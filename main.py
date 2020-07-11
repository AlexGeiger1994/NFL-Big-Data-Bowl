
from src.data.dao.DAO import DAO
from src.pipeline.preprocessing.Preprocessing import Preprocessing
from src.features.position.PositionCount import PositionCount


# setup objects
dao = DAO()
preprocessing = Preprocessing()
positions = PositionCount()


# process
train = dao.getDataset()

train = preprocessing.transform(train)

train = positions.positionCount(train)

print(train.head())




