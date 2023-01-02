import collections
import numpy as np
import pandas as pd
 #Split the subset by rating to create new train, val, and test splits 

by_rating = collections.defaultdict(list)
for _, row in review_subset.iterrows():
    by_rating[row.rating].append(row.to_dict())
# Create split data 
final_list = [] 
np.random.seed(args.seed)
for _, item_list in sorted(by_rating.items()): 
    np.random.shuffle(item_list)
    n_total = len(item_list)
    n_train = int(args.train_proportion * n_total) 
    n_val = int(args.val_proportion * n_total) 
    n_test = int(args.test_proportion * n_total)
# Give data point a split attribute 
    for item in item_list[:n_train]:
        item['split'] = 'train'
    for item in item_list[n_train:n_train+n_val]: 
        item['split'] = 'val'
    for item in item_list[n_train+n_val:n_train+n_val+n_test]: 
        item['split'] = 'test'
    # Add to final list 
    final_list.extend(item_list)
final_reviews = pd.DataFrame(final_list)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text) 
    text = re.sub(r"[ˆa-zA-Z.,!?+", r" ", text) 
    return text
final_reviews.review = final_reviews.review.apply(preprocess_text)

from torch.utils.data import Dataset
class ReviewDataset(Dataset):

# Args:
# review_df (pandas.DataFrame): the dataset
# vectorizer (ReviewVectorizer): vectorizer instantiated from dataset

    def __init__(self, review_df, vectorizer):
        self.review_df = review_df 
        self._vectorizer = vectorizer
        self.train_df = self.review_df[self.review_df.split=='train'] 
        self.train_size = len(self.train_df)
        self.val_df = self.review_df[self.review_df.split=='val'] 
        self.validation_size = len(self.val_df)
        self.test_df = self.review_df[self.review_df.split=='test'] 
        self.test_size = len(self.test_df)
        self._lookup_dict = {'train': (self.train_df, self.train_size), 'val': (self.val_df, self.validation_size),
        'test': (self.test_df, self.test_size)} 
        self.set_split('train')
@classmethod
def load_dataset_and_make_vectorizer(cls, review_csv):
    review_df = pd.read_csv(review_csv)