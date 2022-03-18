
class test():
    def __init__(self):
        self.train_batch = 0
        self.logfile = open("asd.csv", "w")
        pass

    def _implements_train_batch_hooks(self):
        return True

    def _implements_test_batch_hooks(self):
        return True

    def _implements_predict_batch_hooks(self):
        return False

    def set_model(self, a):
        print("Model", a)

    def set_params(self, a):
        print("Params", a)

    def on_train_begin(self, a):
        pass

    def on_train_batch_begin(self, a, b):
        pass

    def on_train_batch_end(self, a, b):
        pass

    def on_train_end(self, a):
        pass

    def on_test_begin(self, a):
        pass

    def on_test_batch_begin(self, a, b):
        pass

    def on_test_batch_end(self, a, b):
        pass

    def on_test_end(self, a):
        pass

    def on_epoch_begin(self, a, b):
        pass

    def on_epoch_end(self, a, b):
        loss = b["loss"]
        accuracy = b["accuracy"]
        val_loss = b["val_loss"]
        val_accuracy = b["val_accuracy"]
        self.logfile.write(f"{a},{loss},{accuracy},{val_loss},{val_accuracy}\n")
