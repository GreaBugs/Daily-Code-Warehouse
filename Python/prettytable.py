from prettytable import PrettyTable

results = PrettyTable()
results.field_names = ["序号", "姓名", "年龄"]

for i in range(4):
    results.add_row(
        [
            "[" + str(int(i)) + "]",
            "荒【{:.2f}%】".format(2 * 100),
            "{}".format(23),
        ]
    )

print(results.get_string())