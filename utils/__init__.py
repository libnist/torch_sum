from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self,
                 path,
                 return_tensors="pt",
                 return_attention_mask=False,
                 skip_special_tokens_in_decode=True):
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.return_tensors = (return_tensors
                               if return_tensors in ["pt", "np", "tf"]
                               else None)

        self.attention_mask = return_attention_mask

        self.skip_special_tokens_in_decode = skip_special_tokens_in_decode

    def __call__(self, inputs):
        result = self.tokenizer(inputs,
                                return_tensors=self.return_tensors,
                                return_attention_mask=self.attention_mask)
        if len(result.keys()) == 1:
            return result[list(result.keys())[0]]
        return result

    def decode(self, inputs):
        if isinstance(inputs[0], list):
            return self.tokenizer.batch_decode(
                inputs,
                skip_special_tokens=self.skip_special_tokens_in_decode
            )
        else:
            return self.tokenizer.decode(
                inputs,
                skip_special_tokens=self.skip_special_tokens_in_decode
            )
