import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


# Read borrowing history
data = pd.read_csv("data/borrowing_history.csv")

# Create encoders
month_encoder = LabelEncoder()
book_encoder = LabelEncoder()

# Convert text to numbers
data["Month"] = month_encoder.fit_transform(data["Month"])
data["Book_Title"] = book_encoder.fit_transform(data["Book_Title"])

# Input feature
X = data[["Month"]]

# Output label
y = data["Book_Title"]

# Train model
model = DecisionTreeClassifier(random_state=42)

model.fit(X, y)


def predict_book(month):

    try:

        month_value = month_encoder.transform([month])

    except ValueError:

        return "Month not found."

    prediction = model.predict([[month_value[0]]])

    book = book_encoder.inverse_transform(prediction)

    return book[0]


if __name__ == "__main__":

    print("====== Book Demand Prediction ======\n")

    while True:

        month = input("Enter Month (or exit): ")

        if month.lower() == "exit":
            break

        result = predict_book(month)

        print("\nPredicted High Demand Book")

        print("--------------------------")

        print(result)