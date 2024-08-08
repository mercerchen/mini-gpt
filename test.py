from train import encode, decode

if __name__ == "__main__":
    assert decode(encode("hello")) == "hello"    