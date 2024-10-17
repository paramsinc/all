from code.config import config
from code.eval import evaluate_model
from code.model import create_model
from code.train import split_data, train_model

if __name__ == "__main__":
    # Create the model
    model = create_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(model.summary())

    # prep the data
    X_train_event, X_test_event, X_train_time, X_test_time, y_train, y_test = (
        split_data()
    )
    # now train the model
    history = train_model(
        model=model,
        X_train_event=X_train_event,
        X_test_event=X_test_event,
        X_train_time=X_train_time,
        X_test_time=X_test_time,
        y_train=y_train,
        y_test=y_test,
    )

    # Evaluate the model
    evaluate_model(
        model=model,
        X_test_event=X_test_event,
        X_test_time=X_test_time,
        y_test=y_test,
        history=history,
    )

    # Save the model
    model.save(config.prod_model_path)

    # TODO store it in GCP