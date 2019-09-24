import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

from dataset import load_dataset_pd

MAX_NUM_SAMPLE = 10
MAX_NUM_WORDS_TO_SHOW = 100


def cmd_print(title: str, value: any):
    side_padding = 2
    roof = "*" * (len(title) + (side_padding * 2) + 4)
    print("\n")
    print(roof)
    print(f"**{' ' * side_padding}{title}{' ' * side_padding}**")
    print(roof)
    if value is not None:
        print(value)


df = load_dataset_pd()

# Info
cmd_print("Info", None)
df.info()

# Sample
# cmd_print(f"Sample of {MAX_NUM_SAMPLE}", df.head(MAX_NUM_SAMPLE))

# All categories
cmd_print("All categories", df["category"].value_counts())

# Box plot of name lengths
df["name_len"] = [len(r) for r in df["name"]]
fig, ax = plt.subplots(figsize=(5, 5))
fig.suptitle("Box plot of name lengths")
plt.boxplot(df["name_len"])
plt.show()


# See most common words
def wordcloud_by_category(category: str):
    names = df[df.category.str.contains(category)]["name"]
    all_names = " ".join(names.str.lower())
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="black", max_words=1000).generate(all_names)
    plt.title = f"Common words for '{category}'"
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


wordcloud_by_category("graphic cards")
wordcloud_by_category("make up")
