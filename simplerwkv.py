import os, math, gc, importlib
import torch
import torch.utils.checkpoint

# SimpleRWKV specific imports
from transformers import PreTrainedTokenizerFast

# Current script dir
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../'))

# SimpleRWKV is a wrapper for RWKV that allows for simple usage of the model
#
# it is not meant to be highly performant, but rather a simple minimal way to run the RWKV trainer module
# in inference mode, and can be used to validate the model trainer code / its changes
class SimpleRWKV():

    def __init__(
            self,
            model,
            args,
            ctx_len:int = 1024,
            device:str = "cuda",
            dtype_str:str = "fp32"
        ):

        self.model = model

        # Log the mismatch dtype
        dtype = torch.float32
        if dtype_str == "16":
            dtype = torch.float16
        elif dtype_str == "bf16":
            dtype = torch.bfloat16
        elif dtype_str == "32":
            dtype = torch.float32
        else:
            print("[SimpleRWKV] Warning: dtype mismatch, only fp16 bf16 fp32 is supported (for now)")

        # Prepare the model config with the model path, and custom torch load
        #model_config = {}
        #model_config["load_model"] = model_path
        #model_config["ctx_len"] = ctx_len

        # FIXME
        #model_config["version"] = "6.0"
        #model_config["strict_loading"] = False
        #model_config["num_experts"] = 8

        # This feature depends on deepspeed
        #model_config["grad_cp"] = False
        # model_config["_torch_load_state"] = loaded_state

        # Save the config settings
        self.ctx_len = ctx_len
        self.device = device

        # Lets actually load the model
        #trainer = Trainer(precision=dtype_str, accelerator='cuda', devices=1)
        #fabric = Lightning.Fabric(precision=dtype_str, accelerator='cuda', devices=1)
        #with fabric.init_module():
        print("dtype of model itself started as ", self.model.ln_out.weight.dtype)

        # Lets map it over to the respective device type
        # and set it to run as eval/inference mode
        print("Desired dtype", dtype)
        self.model.to(dtype)
        self.model.to(device)
        self.model.eval()
        if dtype != torch.float:
            torch.set_autocast_gpu_dtype(dtype)

        print("dtype of model itself became ", self.model.ln_out.weight.dtype)

        # The tokenizer object values
        self.fastTokenizer = None
        self.worldTokenizer = None

        # Setup the tokenizer
        if args.vocab_size == 50277:
            # Use the neox tokenizer
            tokenizer_file = os.path.join(SCRIPT_DIR,"./dataflow/20B_tokenizer.json")
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            self.fastTokenizer = tokenizer
        elif args.vocab_size == 65536:
            # Use the world tokenizer
            from src.dataflow.trie_tokenizer import MT_TRIE_TOKENIZER
            world_tokenizer = MT_TRIE_TOKENIZER(os.path.join(SCRIPT_DIR, "./dataflow/rwkv_vocab_v20230424.txt"))
            self.worldTokenizer = world_tokenizer
        else:
            raise NotImplementedError(f"Unsupported vocab size ({args.vocab_size}) - custom tokenizer not supported")

    # Encoding strings
    def encode(self, text: str):
        if self.worldTokenizer != None:
            return self.worldTokenizer.encode(text)
        return self.fastTokenizer.encode(text)

    # Decoding strings
    def decode(self, tokens: list):
        if self.worldTokenizer != None:
            return self.worldTokenizer.decode(tokens)
        return self.fastTokenizer.decode(tokens)

    # Forwarding logic, withoout torch._no_grad() context
    def _forward(
            self, tokens, 
            stateObj = None,
            all_logits = False
        ):

        logits_arr = None
        token_len = len(tokens)

        # The all_logits array, if requested
        all_logits_arr = None

        # For each token, process the state, in batches up to ctx_len
        for i in range(0, token_len, self.ctx_len):
            # Token set
            token_set = tokens[i:i+self.ctx_len]

            # Check if tokens are already tensors
            batch_tokens = torch.tensor(
                token_set, 
                dtype=torch.long, device=self.device
            ).unsqueeze(0)
            
            # Compute the logits and state
            logits_arr, stateObj = self.model.forward(
                batch_tokens, stateObj
            )

            # Build the all_logits array
            if all_logits:
                if all_logits_arr is None:
                    all_logits_arr = logits_arr[0]
                else:
                    all_logits_arr = torch.cat([all_logits_arr, logits_arr[0]], dim=0)

        # Return the logits and state
        if all_logits:
            return all_logits_arr, stateObj
        else:
            return logits_arr[0][-1], stateObj
    
    # Forwarding logic, with torch._no_grad() context
    def forward(
            self, tokens:list, 
            stateObj = None,
            all_logits = False
        ):
        with torch.no_grad():
            return self._forward(tokens, stateObj, all_logits)

    # Sampling logits
    def sample_logits(
            self, logits, 
            prv_tokens=[0], 
            temperature=1.0, top_p=0.9,
            token_ban: list = []
            ):
        # Copy to CPU first
        logits = logits.float().cpu()

        # Max negative float
        max_neg = -torch.finfo(torch.float).max

        # Apply token ban
        for x in token_ban:
            logits[x] = max_neg
        
        # Remove NaNs from logits
        for x in range(len(logits)):
            if torch.isnan(logits[x]):
                logits[x] = max_neg

        # Handle sampling with temperature
        if temperature > 0.0:
            probs = F.softmax(logits, dim=-1)
            sorted_probs = torch.sort(probs, descending=True)[0]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).float().cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if temperature != 1.0:
                probs = probs.pow(1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return out
        else: 
            # Since the tokenizer sample does not support temp==0
            # we handle this case ourself, by fining the top token
            return torch.argmax(logits, dim=-1).item()

    # Completion API
    def completion(self, 
            prompt, 
            max_tokens: int = 32,
            temperature: float = 1.0,
            top_p: float = 0.9,
            token_ban: list = [],
            start_state = None,
            stream_to_stdout: bool = False,
        ):
        # Encode the context, if its a string
        if isinstance(prompt, str):
            enc = self.encode(prompt)
        # Check if the prompt is a list of tokens
        elif isinstance(prompt, list):
            enc = prompt
        else:
            raise ValueError("Prompt must be a string or a list of tokens")

        # Keep track of the logits and state
        logits = None
        stateObj = start_state

        # For each token, process the state
        logits, stateObj = self.forward(enc, stateObj)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Generate each token
        out_tokens = []
        for i in range(max_tokens):
            ttt = self.sample_logits(
                logits, 
                # prv_tokens=full_tokens,
                temperature=temperature, top_p=top_p,
                token_ban=token_ban
            )
            
            # Append the token
            out_tokens.append(ttt)
            # full_tokens.append(ttt)
            if stream_to_stdout:
                print(self.decode([ttt]), end="", flush=True)

            # Perform the forward pass
            logits, stateObj = self.forward([ttt], stateObj)

        # Decode the tokens
        out_str = self.decode(out_tokens)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Return the output string, and state
        return out_str, stateObj
