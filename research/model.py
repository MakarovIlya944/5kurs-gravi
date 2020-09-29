
class Model():
  """
  Base class for model entity (aka IModel)
  """
  name = "base"

  """
  Learn model to ./models/<name>/models/<datetime>
  """
  def learn(self, input, output):
    pass

  """
  Predict from ./models/<name>/input/ to ./models/<name>/output/
  """
  def predict(self):
    pass