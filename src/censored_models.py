import numpy as np

class TwoStageCensoredModel:
    """
    Implementa um Modelo de Hurdle (Duas Etapas) para lidar com a censura no piso (ex: PLD).
    Etapa 1: Classificador prevê a probabilidade de a variável estar no piso.
    Etapa 2: Regressor prevê o valor contínuo da variável, treinado apenas nos casos em que não está no piso.
    
    A previsão final é o valor esperado: E[Y] = P(Piso)*Piso + (1 - P(Piso))*Y_regressor.
    """
    def __init__(self, classifier, regressor, target_is_log=False):
        self.classifier = classifier
        self.regressor = regressor
        self.target_is_log = target_is_log
        
    def fit(self, X, y, piso_array):
        """
        piso_array: array com os valores do piso correspondentes a cada amostra de X
        """
        # Se a variável y está em log, precisamos reverter para comparar com o piso
        if self.target_is_log:
            is_piso = (np.exp(y) <= piso_array + 0.01).astype(int)
        else:
            is_piso = (y <= piso_array + 0.01).astype(int)
        
        # Treina o classificador
        self.classifier.fit(X, is_piso)
        
        # Filtra os dados onde o valor NÃO está no piso para treinar o regressor
        mask_reg = (is_piso == 0)
        
        # Se houver dados suficientes fora do piso, treina o regressor
        if mask_reg.sum() > 0:
            self.regressor.fit(X[mask_reg], y[mask_reg])
        else:
            self.regressor.fit(X, y)
            
        return self

    def predict(self, X, piso_array, teto_array):
        """
        piso_array, teto_array: arrays com os valores correspondentes de piso e teto para X
        """
        if self.target_is_log:
            piso_target = np.log(piso_array)
            teto_target = np.log(teto_array)
        else:
            piso_target = piso_array
            teto_target = teto_array
        
        if hasattr(self.classifier, "predict_proba"):
            p_piso = self.classifier.predict_proba(X)[:, 1]
        elif hasattr(self.classifier, "decision_function"):
            # Para RidgeClassifier, podemos usar função sigmoide no decision_function para simular predict_proba
            df = self.classifier.decision_function(X)
            # Sigmoid
            p_piso = 1 / (1 + np.exp(-df))
        else:
            p_piso = self.classifier.predict(X)
            
        y_reg_pred = self.regressor.predict(X)
        y_reg_pred = np.clip(y_reg_pred, piso_target, teto_target)
        
        y_pred = p_piso * piso_target + (1 - p_piso) * y_reg_pred
        return y_pred

