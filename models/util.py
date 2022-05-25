from models import mnist_lenet, cifar_lenet, cifar_resnet


def get_model(model_name: str, dataset_name: str, device: str = "cpu"):
    # ResNet
    if hasattr(cifar_resnet, model_name):
        if dataset_name == 'CIFAR10':
            model = getattr(cifar_resnet, model_name)().to(device)
            model.apply(cifar_resnet.Model.initialize)
        else:
            raise NotImplementedError()
    # LeNet
    else:
        if dataset_name == 'MNIST':
            model = mnist_lenet.Model().to(device)
            model.apply(mnist_lenet.Model.initialize)
        elif dataset_name == 'CIFAR10':
            model = cifar_lenet.Model().to(device)
            model.apply(cifar_lenet.Model.initialize)
        else:
            raise NotImplementedError()
    return model
