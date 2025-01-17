from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased')

print(unmasker("J'ai un oreiller noir. mon oreiller est [MASK]."))
# premier résultat :
# {'score': 0.13341058790683746, 'token': 15587, 'token_str': 'noir', 'sequence': "j ' ai un oreiller noir. mon oreiller est noir."}

