import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms._transforms_video as VT

import time

def video_collate_fn(batch):
    """
    Custom collate function for a batch of videos.

    Args:
        batch (list of tuples): Each element is a tuple where the first element is a video tensor
                                and the second element is the label (optional).

    Returns:
        videos (torch.Tensor): Padded batch of videos.
        labels (torch.Tensor): Batch of labels.
    """

    # Separate videos and labels
    videos, audio, labels = zip(*batch)

    # Get the maximum number of frames (T) in a video in the batch
    max_frames = max(video.shape[0] for video in videos)

    # Pad all videos to have the same number of frames (max_frames)
    padded_videos = []
    for video in videos:
        # Get the padding size
        padding_size = max_frames - video.shape[0]
        # Create a padding tensor of zeros with the same C, H, W as the original video
        padding = torch.zeros(
            (padding_size, video.shape[1], video.shape[2], video.shape[3])
        )
        # Concatenate the video with the padding along the time dimension (T)
        padded_video = torch.cat((video, padding), dim=0)
        padded_videos.append(padded_video)

    # Stack all padded videos into a single tensor
    videos_tensor = torch.stack(padded_videos, dim=0)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(labels)

    return videos_tensor, labels_tensor


if __name__ == "__main__":
    # Define the transform to apply to the video frames
    transform_train = transforms.Compose(
        [
            T.Resize((200, 200)),
            VT.RandomHorizontalFlipVideo(),
            T.ConvertImageDtype(torch.float32),
            # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
            T.RandomCrop((172, 172)),
        ]
    )

    transform_test = transforms.Compose(
        [
            T.Resize((200, 200)),
            T.ConvertImageDtype(torch.float32),
            # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
            T.CenterCrop((172, 172)),
        ]
    )

    # Load the HMDB51 dataset
    train_dataset = torchvision.datasets.HMDB51(
        root="./data/hmdb51/hmdb51_org",
        annotation_path="./data/hmdb51/splits",
        frames_per_clip=16,
        frame_rate=5,
        step_between_clips=2,
        train=True,
        num_workers=4,
        transform=transform_train,
        output_format="TCHW",
    )

    test_dataset = torchvision.datasets.HMDB51(
        root="./data/hmdb51/hmdb51_org",
        annotation_path="./data/hmdb51/splits",
        frames_per_clip=16,
        frame_rate=5,
        step_between_clips=2,
        train=False,
        num_workers=4,
        transform=transform_test,
        output_format="TCHW",
    )

    print("Number of training videos:", len(train_dataset))
    print("Number of test videos:", len(test_dataset))
    print("Number of classes:", len(train_dataset.classes))
    print(train_dataset[0][0].shape)
    print(train_dataset[0][1].shape)
    # # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=video_collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=video_collate_fn, pin_memory=True
    )

    # print(train_loader[0][0].shape)
    # print(train_loader[0][1].shape)
    N_EPOCHS = 1
    start_time = time.time()

    # Load the I3D model
    i3d = torchvision.models.video.r3d_18(pretrained=True)  # Load the pretrained I3D model

    # Replace the last layer to match the number of classes in HMDB51 (51 classes)
    i3d.fc = torch.nn.Linear(i3d.fc.in_features, 51)

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i3d = i3d.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(i3d.parameters(), lr=0.001)

    # Training loop
    for epoch in range(N_EPOCHS):
        i3d.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = i3d(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f'Epoch [{epoch+1}/{N_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        # Evaluation loop
        i3d.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = i3d(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Execution time:", "{:5.2f}".format(time.time() - start_time), "seconds")


    print('Finished Training')