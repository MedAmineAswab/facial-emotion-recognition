
def split_data(X, Y, df):
    """
    Split les données en train / validation / test basé sur la colonne 'Usage' du DataFrame.

    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    """

    # ======================
    # Train set
    # ======================
    train_idx = df['Usage'] == 'Training'
    X_train = X[train_idx]
    Y_train = Y[train_idx]

    # ======================
    # Validation set
    # ======================
    val_idx = df['Usage'] == 'PublicTest'
    X_val = X[val_idx]
    Y_val = Y[val_idx]

    # ======================
    # Test set
    # ======================
    test_idx = df['Usage'] == 'PrivateTest'
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    # ======================
    # Vérification des dimensions 
    # ======================
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_val:", X_val.shape, "Y_val:", Y_val.shape)
    print("X_test:", X_test.shape, "Y_test:", Y_test.shape)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test
  