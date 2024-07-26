import torch
import torchvision
import time
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms._transforms_video as VT
import torch.nn.functional as F
import torch.optim as optim

from movinets import MoViNet
from movinets.config import _C


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


def train_iter(model, optimz, data_load, loss_val):
    samples = len(data_load.dataset)
    model.train()
    model.cuda()
    model.clean_activation_buffers()
    optimz.zero_grad()
    for i, (data, target) in enumerate(data_load):
        out = F.log_softmax(model(data.cuda()), dim=1)
        loss = F.nll_loss(out, target.cuda())
        loss.backward()
        optimz.step()
        optimz.zero_grad()
        model.clean_activation_buffers()
        if i % 50 == 0:
            print(
                "["
                + "{:5}".format(i * len(data))
                + "/"
                + "{:5}".format(samples)
                + " ("
                + "{:3.0f}".format(100 * i / len(data_load))
                + "%)]  Loss: "
                + "{:6.4f}".format(loss.item())
            )
            loss_val.append(loss.item())


def evaluate(model, data_load, loss_val):
    model.eval()

    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    model.clean_activation_buffers()
    with torch.no_grad():
        for data, target in data_load:
            output = F.log_softmax(model(data.cuda()), dim=1)
            loss = F.nll_loss(output, target.cuda(), reduction="sum")
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            csamp += pred.eq(target.cuda()).sum()
            model.clean_activation_buffers()
    aloss = tloss / samples
    loss_val.append(aloss)
    print(
        "\nAverage test loss: "
        + "{:.4f}".format(aloss)
        + "  Accuracy:"
        + "{:5}".format(csamp)
        + "/"
        + "{:5}".format(samples)
        + " ("
        + "{:4.2f}".format(100.0 * csamp / samples)
        + "%)\n"
    )


def train_iter_stream(model, optimz, data_load, loss_val, n_clips=2, n_clip_frames=8):
    """
    In causal mode with stream buffer a single video is fed to the network
    using subclips of lenght n_clip_frames.
    n_clips*n_clip_frames should be equal to the total number of frames presents
    in the video.

    n_clips : number of clips that are used
    n_clip_frames : number of frame contained in each clip
    """
    
    # TODO: Apply automatic mixed precision training.
    # clean the buffer of activations
    samples = len(data_load.dataset)
    model.cuda()
    model.train()
    model.clean_activation_buffers()
    optimz.zero_grad()

    scaler = torch.amp.GradScaler()
    
    for i, (data, target) in enumerate(data_load):
        data = data.cuda()
        target = target.cuda()
        l_batch = 0

        # rearrange the data to b c t h w
        data = data.permute(0, 2, 1, 3, 4)

        # backward pass for each clip
        for j in range(n_clips):
            with torch.amp.autocast('cuda'):
                output = F.log_softmax(
                    model(
                        data[:, :, (n_clip_frames) * (j) : (n_clip_frames) * (j + 1)]
                    ),
                    dim=1,
                )
                loss = F.nll_loss(output, target)
                _, pred = torch.max(output, dim=1)
                loss = F.nll_loss(output, target) / n_clips
            
            scaler.scale(loss).backward()
            l_batch += loss.item() * n_clips
            
        scaler.step(optimz)
        scaler.update()
        optimz.zero_grad()

        # clean the buffer of activations
        model.clean_activation_buffers()
        if i % 50 == 0:
            print(
                "["
                + "{:5}".format(i * len(data))
                + "/"
                + "{:5}".format(samples)
                + " ("
                + "{:3.0f}".format(100 * i / len(data_load))
                + "%)]  Loss: "
                + "{:6.4f}".format(l_batch)
            )
            loss_val.append(l_batch)


def evaluate_stream(model, data_load, loss_val, n_clips=2, n_clip_frames=8):
    model.eval()
    model.cuda()
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    with torch.no_grad():
        for data, target in data_load:
            data = data.cuda()
            target = target.cuda()
            
            # rearrange the data to b c t h w
            data = data.permute(0, 2, 1, 3, 4)
            
            model.clean_activation_buffers()
            for j in range(n_clips):
                output = F.log_softmax(
                    model(
                        data[:, :, (n_clip_frames) * (j) : (n_clip_frames) * (j + 1)]
                    ),
                    dim=1,
                )
                loss = F.nll_loss(output, target)
            _, pred = torch.max(output, dim=1)
            tloss += loss.item()
            csamp += pred.eq(target).sum()

    aloss = tloss / len(data_load)
    loss_val.append(aloss)
    print(
        "\nAverage test loss: "
        + "{:.4f}".format(aloss)
        + "  Accuracy:"
        + "{:5}".format(csamp)
        + "/"
        + "{:5}".format(samples)
        + " ("
        + "{:4.2f}".format(100.0 * csamp / samples)
        + "%)\n"
    )


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

    model = MoViNet(_C.MODEL.MoViNetA0, causal=True, pretrained=True)
    start_time = time.time()

    trloss_val, tsloss_val = [], []
    model.classifier[3] = torch.nn.Conv3d(2048, 51, (1, 1, 1))
    optimz = optim.Adam(model.parameters(), lr=0.00005)
    for epoch in range(1, N_EPOCHS + 1):
        print("Epoch:", epoch)
        train_iter_stream(model, optimz, train_loader, trloss_val)
        evaluate_stream(model, test_loader, tsloss_val)

    print("Execution time:", "{:5.2f}".format(time.time() - start_time), "seconds")
