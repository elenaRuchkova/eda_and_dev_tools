import pandas as pd
import joblib
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

X_test_transformed = pd.read_csv(r'.\models\X_test_transformed.csv')
y_test = pd.read_csv(r'.\models\y_test.csv')
model = joblib.load(r'.\models\best_model.pkl')

if __name__ == '__main__':

    explainer = ClassifierExplainer(model, X_test_transformed, y_test)
    #explainer.shap_interaction_values(X_test_transformed)
    db = ExplainerDashboard(explainer)
    #explainer.dump("explainer.joblib")
    joblib.dump(explainer, '.\obj\explainer.joblib')
    db.to_yaml(r'.\obj\dashboard.yaml', explainerfile='.\obj\explainer.joblib', dump_explainer=True)#