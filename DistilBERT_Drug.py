import pandas as pd
data = pd.read_csv("DrugTokens.csv",encoding="latin1" )

data.head(50)

data =data.fillna(method ="ffill")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data["Sentence #"] = LabelEncoder().fit_transform(data["Sentence #"] )

data.rename(columns={"Sentence #":"sentence_id","Word":"words","Tag":"labels"}, inplace =True)

data["labels"] = data["labels"].str.upper()

X= data[["sentence_id","words"]]
Y =data["labels"]

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size =0.2)

train_data = pd.DataFrame({"sentence_id":x_train["sentence_id"],"words":x_train["words"],"labels":y_train})
test_data = pd.DataFrame({"sentence_id":x_test["sentence_id"],"words":x_test["words"],"labels":y_test})

train_data

!pip install simpletransformers

from simpletransformers.ner import NERModel,NERArgs

label = data["labels"].unique().tolist()
label

args = NERArgs()
args.num_train_epochs = 1
args.learning_rate = 1e-4
args.overwrite_output_dir =True
args.train_batch_size = 32
args.eval_batch_size = 32

model = NERModel('distilbert', 'distilbert-base-cased',labels=label,args =args, use_cuda = False)

model.train_model(train_data,eval_data = test_data,acc=accuracy_score)

result, model_outputs, preds_list = model.eval_model(test_data)

result

prediction, model_output = model.predict(["- continue Cosopt BID, Latanoprost qhs take Refresh"])
prediction
