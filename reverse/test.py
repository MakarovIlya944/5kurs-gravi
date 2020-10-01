from builder import *
import unittest
from numpy import array

defNet = {
    'count': (5,1,5),
    'right': (3000,50,-1500),
    'left': (1000,0,-500)
  }

class TestNetBuilder(unittest.TestCase):

  def base_test(self, defNet):
    params = defNet
    params['c_value'] = self.c_value
    params['width'] = self.width
    params['center'] = self.center
    inds = self.center - self.width
    inds_l = []
    for ind,i in enumerate(inds):
      if i < 0:
        inds_l.append(0)
        continue
      if i >= defNet['count'][ind]:
        inds_l.append(defNet['count'][ind]-1)
        continue
      inds_l.append(i)

    inds = self.center + self.width
    inds_r = []
    for ind,i in enumerate(inds):
      if i < 0:
        inds_r.append(0)
        continue
      if i >= defNet['count'][ind]:
        inds_r.append(defNet['count'][ind]-1)
        continue
      if inds_l[ind] > i:
        inds_r.append(inds_l[ind])
        continue
      inds_r.append(i)

    net = center_build(params=params)
    vals = [
      [i for i in range(inds_l[c_i], inds_r[c_i]+1)]
      for c_i in range(3)
    ]

    values = []
    for x in vals[0]:
      for y in vals[1]:
        for z in vals[2]:
          values.append((x,y,z))
    for k in values:
      self.assertEqual(net[k], self.c_value)

  def test_center_thin(self):
    self.c_value = 10
    self.center = array([2,0,2])
    self.width = array([1,0,1])
    global defNet
    self.base_test(defNet)

  def test_center_border(self):
    self.c_value = 10
    self.center = array([0,0,0])
    self.width = array([1,0,1])
    global defNet
    self.base_test(defNet)

if __name__ == '__main__':
    unittest.main()