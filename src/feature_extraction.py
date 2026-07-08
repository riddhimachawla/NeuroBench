from mne.decoding import CSP


class CSPExtractor:

    def __init__(self, n_components=8):

        self.csp = CSP(
            n_components=n_components,
            log=True,
            norm_trace=False
        )

    def fit_transform(self, X, y):

        return self.csp.fit_transform(X, y)

    def transform(self, X):

        return self.csp.transform(X)