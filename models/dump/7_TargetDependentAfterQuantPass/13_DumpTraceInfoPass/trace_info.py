from typing import List, Tuple

class DataItem:
    number: int
    name: int
    exec_time: float
    type: bool
    begin: float
    end: float
    def __init__(self, number, name, type, begin, end, exec_time) -> None:
        self.number = number
        self.name = name
        self.type = type
        self.begin = begin
        self.end = end
        self.exec_time = exec_time

class VersionData:
  name: str
  data: DataItem

  def __init__(self, name, data) -> None:
    self.name = name
    self.data = data

  def print(self):
    print(f'| number | name | type | begin | end | exec |')
    print(f'|-|-|-|-|-|-|')
    time = 0
    for p in self.data:
      print(f'| {p.number} | {p.name} | {p.type} | {p.begin} | {p.end} |  {p.exec_time} |')
      time += p.exec_time
    print(f'| total |  | {time} | | |')
v0 = VersionData("v0",[
DataItem(0, "TileLayerGroup_10", False, 0.0, 0.0, 0.0),
DataItem(1, "TileLayerGroup_9", False, 0.0, 0.0, 0.0),
DataItem(2, "TileLayerGroup_8", False, 0.0, 0.0, 0.0),
DataItem(3, "TileLayerGroup_7", False, 0.0, 0.0, 0.0),
DataItem(4, "TileLayerGroup_6", False, 0.0, 0.0, 0.0),
DataItem(5, "TileLayerGroup_5", False, 0.0, 0.0, 0.0),
DataItem(6, "TileLayerGroup_4", False, 0.0, 0.0, 0.0),
DataItem(7, "TileLayerGroup_3", False, 0.0, 0.0, 0.0),
DataItem(8, "TileLayerGroup_2", False, 0.0, 0.0, 0.0),
DataItem(9, "TileLayerGroup_1", False, 0.0, 0.0, 0.0),
DataItem(10, "TileLayerGroup_0", False, 0.0, 0.0, 0.0),
])
if __name__ == '__main__':
  v0.print()
