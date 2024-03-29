class TimeStamp:
    def __init__(self, minute: int, second: int, hour: int = 0, millisecond: int = 0):
        self.minute = minute
        self.second = second
        self.hour = hour
        self.millisecond = millisecond

    def to_seconds(self) -> int:
        return self.hour * 3600 + self.minute * 60 + self.second

    def __sub__(self, other):
        return TimeStamp(minute=self.minute - other.minute, second=self.second - other.second, hour=self.hour - other.hour, millisecond=self.millisecond - other.millisecond)

config_rigid_220 = [
    {
        'path': "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/ColorizedRigid/2-20-1.mov",
        'clips': [
            (TimeStamp(0,2), TimeStamp(0, 18)),
            (TimeStamp(0, 52), TimeStamp(1, 4)),  
        ]
    },
    {
        'path': "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/ColorizedRigid/2-20-2.mov",
        'clips': [
            (TimeStamp(0, 3), TimeStamp(0, 26)),
            (TimeStamp(0, 38), TimeStamp(0, 54)),
        ]
    },
    {
        'path': "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/ColorizedRigid/2-20-3.mov",
        'clips': [
            (TimeStamp(0, 3), TimeStamp(1, 8)),
        ]
    }
]


config_rigid_1121 = [
    {
        'path': "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/11-21-1.mov",
        'clips': [
            (TimeStamp(5, 41), TimeStamp(7, 32)),
            (TimeStamp(3, 26), TimeStamp(4, 23)),
            (TimeStamp(2, 19), TimeStamp(3, 32)),
        ]
    },
    {
        'path': "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/11-21-2.mov",
        'clips': [
            (TimeStamp(0, 0), TimeStamp(0, 21)),
            (TimeStamp(0, 30), TimeStamp(0, 51), 'Wire is going over the vessel'),
        ]
    },
    {
        'path': "/mnt/e/Workspace/CathederTelesurgery/Data/Videos/11-21-3.mov",
        'clips': [
            (TimeStamp(0, 5), TimeStamp(0, 59), 'Overlapping could cause loose of tracking'),
        ]
    }
]


video_sample_config = config_rigid_220