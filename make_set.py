for k in range(1, 11):
    with open(f"{k}.txt", "r") as file:
        expressions = file.readlines()
    old_len = len(expressions)
    expressions = set(expressions)
    new_len = len(expressions)
    print(f"reduced by {((old_len - new_len) / old_len) * 100} percent")
    with open(f"data/new_synthetic/{k}.txt", "w") as file:
        for e in expressions:
            file.write(e)
