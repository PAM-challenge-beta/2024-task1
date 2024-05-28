import torch.nn as nn
import torch.nn.functional as F
import torch

default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path, device=default_device):
    """
    Loads a pre-trained CNN model from a file.

    Args:
        model_path (str): Path to the model file.

    Returns:
        torch model: Loaded CNN model.
    """
    model = CNNModel(device = device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

class CNNModel(nn.Module):
    """
    A simple CNN model for binary classification.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of classes for the output layer.
        
    Returns:
        torch model: Compiled CNN model.
    """
    def __init__(self, input_shape=(128,128,1), num_classes=2, device = "cpu"):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(256)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.flatten = nn.Flatten()
        img_height = input_shape[0]
        self.fc1 = nn.Linear(128 * ((img_height//4 - 2)**2), 64)  # Adjusted to match the flattened dimension
        self.bn3 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)

        self.device = device

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)

        # x = F.softmax(, dim=1)
        return x
    
    def save(self, model_path):
        """
        Saves the model to the specified path.

        Args:
            model_path (str): Path to save the model.
        """
        torch.save(self.state_dict(), f"{model_path}/model.pth")

    def make_prediction(self, x):
        """
        Predicts the class of the input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Predicted class probabilities.
        """
        self.eval()
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            x = self(x)
        return x

# import argparse
# from pathlib import Path
# import torch_dataloader

# def main():
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('hdf5_db', type=str, help='HDF5 Database file path')
#     parser.add_argument('--train_table', default='/train', type=str, help="The table within the hdf5 database where the training data is stored.")
#     parser.add_argument('--val_table', default=None, type=str, help="The table within the hdf5 database where the validation data is stored.")
#     parser.add_argument('--epochs', default=20, type=int, help='The number of epochs')
#     parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
#     parser.add_argument('--output_folder', default=None, type=str, help='Output directory')
#     parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')

#     args = parser.parse_args()
    
#     if args.seed:
#         torch.manual_seed(args.seed)

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # transform = transforms.Compose([
#     #     transforms.ToTensor(),
#     # ])

#     train_loader = torch_dataloader.get_HDF5_dataloader(args.hdf5_db, args.train_table, args.batch_size, 2, transform=None)
#     # val_loader = None
#     # if args.val_table:
#     #     val_loader = torch_dataloader.get_HDF5_dataloader(args.hdf5_db, args.val_table, args.batch_size, 2, transform=None)

#     model = CNNModel(num_classes=2)
#     val_loader = None
#     model = train_model(model, train_loader, val_loader, args.epochs, device)
    
#     if args.output_folder is None:
#         args.output_folder = Path('.').resolve()
#     else:
#         args.output_folder = Path(args.output_folder).resolve()

#     torch.save(model.state_dict(), args.output_folder / 'model.pth')
#     print(f"Model saved to {str(args.output_folder)}")

# if __name__ == "__main__":
#     main()
