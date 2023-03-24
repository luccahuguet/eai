# Load a dataset
from sklearn.datasets import load_iris
from pymfe.mfe import MFE


#  pretty print the feature names and values
def pretty_print(ft):
    ft = dict(zip(ft[0], ft[1]))
    for k, v in ft.items():
        # print every feature name and its value with syntax highlight
        print("\033[1m\033[94m{}\033[0m: \033[1m\033[92m{}\033[0m".format(k, v))
    print("\n\n")


print("\n\n \033[1m\033[94mMFE\033[0m: \033[1m\033[92mMultivariate Feature Extraction\033[0m \n\n")


data = load_iris()
y = data.target
X = data.data

# # Extract default measures

# mfe = MFE()
# mfe.fit(X, y)
# ft = mfe.extract()
# print("Default measures: ")
# pretty_print(ft)

# # Extract general, statistical and information-theoretic measures

# mfe = MFE(groups=["general", "statistical", "info-theory"])
# mfe.fit(X, y)
# ft = mfe.extract()
# print("General, statistical and information-theoretic measures: ")
# pretty_print(ft)

# Extract all available measures

mfe = MFE(groups="all")
mfe.fit(X, y)
ft = mfe.extract()
print("All available measures: ")
pretty_print(ft)

# model_groups = MFE.valid_groups()
# print(model_groups)
