
from src.data.dao.DAO import DAO
from src.pipeline.preprocessing.Preprocessing import Preprocessing
from src.features.position.PositionCount import PositionCount
from src.features.orientation.Orientation import Orientation 


# setup objects
dao = DAO()
preprocessing = Preprocessing()
positions = PositionCount()
orientation = Orientation()

# get dataset
train = dao.getDataset()

# setup pre-processing transformation
train = preprocessing.transform(train)

# get position count
positions.positionCount(train)

# calculate new yardline
train = orientation.update(train)


print(train.head())




