from data_pipeline.Tokenizer import Tokenizer, ChatFormat


def get_tokenizer(tokenizer_path:str):
    tokenizer = Tokenizer(tokenizer_path)
    chat_format = ChatFormat(tokenizer)
    return tokenizer, chat_format