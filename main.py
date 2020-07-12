
from src.data.dao.DAO import DAO
from src.pipeline.preprocessing.Preprocessing import Preprocessing
from src.features.orientation.Orientation import Orientation 
from src.numeric.NumericalMethods import NumericalMethods 


from src.features.player.PositionMeta import PositionMeta
from src.features.player.Offense import Offense 
from src.features.player.Defense import Defense 
from src.features.player.RunningBack import RunningBack 



# setup objects
dao = DAO()
preprocessing = Preprocessing()
positions     = PositionMeta()
orientation   = Orientation()
defense       = Defense()
offense       = Offense()
runningBack   = RunningBack()

print('extract data')

# get dataset
train = dao.getDataset()

print('preprocess data')

# setup pre-processing transformation
train = preprocessing.transform(train)

# get position count
positions.positionCount(train)

# calculate new yardline
train = orientation.update(train)

# forecast defense positions
defenseForecast = defense.Superposition(train)


# Distance Features
offenseI   = offense.forecast(train)
offenseII  = offense.forecast(train,prefix='alpha1',alpha=1, addXY=False,dt_array = [.8])
offenseIII = offense.forecast(train,prefix='alpha7',alpha=7, addXY=False,dt_array = [.2])
    
# build back features
relativityFeatures = runningBack.relativityTheory(train)
offenseForecast    = defense.forecast(train)
defenseFeatures    = defense.defenseFeatures(train)

print(train.head())




