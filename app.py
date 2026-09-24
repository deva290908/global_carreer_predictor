from flask import Flask, render_template, request
from werkzeug.exceptions import HTTPException

import ml_config
import predict

app = Flask(__name__)

# Load the trained models ONCE, when Flask starts (never per request)
predict.load_models()


def page(**kwargs):
    """Render index.html with everything the form needs."""
    return render_template(
        "index.html",
        ready=predict.is_ready(),
        setup_error=predict.get_error(),
        meta=predict.get_metadata(),
        fields=ml_config.FORM_FIELDS,
        labels=ml_config.FEATURE_LABELS,
        categorical=ml_config.CATEGORICAL_FEATURES,
        integer_fields=ml_config.INTEGER_FEATURES,
        currency=ml_config.CURRENCY_SYMBOL,
        **kwargs,
    )


@app.route("/", methods=["GET", "POST"])
def home():
    if not predict.is_ready():
        return page(result=None, form={})

    if request.method == "POST":
        form = request.form.to_dict()
        student, errors = predict.validate_input(form)
        if errors:
            return page(result=None, form=form, errors=errors)

        try:
            result = predict.predict_student_outcome(student)
        except Exception:
            app.logger.exception("Prediction failed")
            return page(result=None, form=form,
                        errors=["Something went wrong while predicting. Please check your inputs and try again."])

        return page(result=result, form=form, student=student)

    return page(result=None, form={})


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    if isinstance(exc, HTTPException):
        return exc
    app.logger.exception("Unexpected error")
    return "Something went wrong. Please go back and try again.", 500


if __name__ == "__main__":
    app.run(debug=False)
