from transformers import BertTokenizer


class chartokenizer(BertTokenizer):
    def _tokenize(self, text):
        split_tokens = []
        # if self.do_basic_tokenize:
        #     for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

        #         # If the token is part of the never_split set
        #         if token in self.basic_tokenizer.never_split:
        #             split_tokens.append(token)
        #         else:
        #             split_tokens += self.wordpiece_tokenizer.tokenize(token)
        # else:
        #     split_tokens = self.wordpiece_tokenizer.tokenize(text)
        for token in text:
            split_tokens.append(token)
        return split_tokens

