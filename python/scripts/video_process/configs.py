class TimeStamp:
    def __init__(self, minute: int, second: int, hour: int = 0, millisecond: int = 0):
        self.minute = minute
        self.second = second
        self.hour = hour
        self.millisecond = millisecond

    def to_seconds(self) -> int:
        return self.hour * 3600 + self.minute * 60 + self.second

    def __sub__(self, other):
        return TimeStamp(
            minute=self.minute - other.minute,
            second=self.second - other.second,
            hour=self.hour - other.hour,
            millisecond=self.millisecond - other.millisecond,
        )


config_aruco_628 = [
    {
        "path": "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/2024-6-28/1.mov",
        "clips": [
            (TimeStamp(0, 31), TimeStamp(0, 44), "ARUCO 4x4.\n This clip has a lot of motion blur, but the aruco markers are still visible"),
            (TimeStamp(0, 55), TimeStamp(1, 10), "ARUCO 4x4.\n This clip has a lot of motion blur, but the aruco markers are still visible"),
        ],
        
    },
    {
        "path": "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/2024-6-28/2.mov",
        "clips": [
            (TimeStamp(0, 6), TimeStamp(0, 19 ), "AprilTag 16h5.\n This clip rotates slower thus fewer motion blur"),
            (TimeStamp(0, 31), TimeStamp(1, 11), "AprilTag 16h5.\n This clip rotates slower thus fewer motion blur"),
        ],
    },
]
output_dir_628 = "/mnt/e/Workspace/CathederTelesurgery/Data/Datasets/ArucoVideo-6-28/"


config_hand_517 = [
    {
        "path": "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/2024-5-17/5-17-2.mov",
        "clips": [
            (TimeStamp(0, 1), TimeStamp(0, 27)),
        ],
    },
    {
        "path": "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/2024-5-17/5-17-3.mov",
        "clips": [
            (TimeStamp(0, 10), TimeStamp(0, 21), "Translation Only"),
            (TimeStamp(0, 23), TimeStamp(0, 30), "Rotation Only"),
            (TimeStamp(1, 2), TimeStamp(1, 20), "Rotation + Translation"),
        ],
    },
]
output_dir_517 = (
    "/mnt/e/Workspace/CathederTelesurgery/Data/Datasets/HandModelVideo-5-17/"
)


config_rigid_220 = [
    {
        "path": "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/ColorizedRigid/2-20-1.mov",
        "clips": [
            (TimeStamp(0, 2), TimeStamp(0, 18)),
            (TimeStamp(0, 52), TimeStamp(1, 4)),
        ],
    },
    {
        "path": "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/ColorizedRigid/2-20-2.mov",
        "clips": [
            (TimeStamp(0, 3), TimeStamp(0, 26)),
            (TimeStamp(0, 38), TimeStamp(0, 54)),
        ],
    },
    {
        "path": "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/ColorizedRigid/2-20-3.mov",
        "clips": [
            (TimeStamp(0, 3), TimeStamp(1, 8)),
        ],
    },
]
output_dir_220 = (
    "/mnt/e/Workspace/CathederTelesurgery/Data/Datasets/RigidModelVideo-2-20/"
)


config_rigid_1121 = [
    {
        "path": "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/11-21-1.mov",
        "clips": [
            (TimeStamp(5, 41), TimeStamp(7, 32)),
            (TimeStamp(3, 26), TimeStamp(4, 23)),
            (TimeStamp(2, 19), TimeStamp(3, 32)),
        ],
    },
    {
        "path": "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/11-21-2.mov",
        "clips": [
            (TimeStamp(0, 0), TimeStamp(0, 21)),
            (TimeStamp(0, 30), TimeStamp(0, 51), "Wire is going over the vessel"),
        ],
    },
    {
        "path": "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/11-21-3.mov",
        "clips": [
            (
                TimeStamp(0, 5),
                TimeStamp(0, 59),
                "Overlapping could cause loose of tracking",
            ),
        ],
    },
]
output_dir_1121 = (
    "/mnt/e/Workspace/CathederTelesurgery/Data/Datasets/RigidModelVideo-11-21/"
)


output_dir = output_dir_628
video_sample_config = config_aruco_628
