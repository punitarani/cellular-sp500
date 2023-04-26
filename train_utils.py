def create_and_train_models(stock_A, stock_B, stock_A_symbol, stock_B_symbol, results_queue):
    model_key, (models_A, models_B) = create_and_train_models(stock_A, stock_B, stock_A_symbol, stock_B_symbol)
    save_model_and_scaler(models_A[0], models_A[1], f"model_A_{stock_A_symbol}-{stock_B_symbol}")
    save_model_and_scaler(models_B[0], models_B[1], f"model_B_{stock_A_symbol}-{stock_B_symbol}")
    results_queue.put((model_key, (models_A, models_B)))
