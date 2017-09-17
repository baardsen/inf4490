import unittest
import oblig1

class TestOblig1Methods(unittest.TestCase):

	def test_pmx_child1(self):
		mother = ['e', 'a', 'b', 'c', 'd']
		father = ['e', 'b', 'a', 'd', 'c']
		
		child = oblig1.pmx_child(mother, father, 2, 4)
		self.assertEqual(['e', 'a', 'b', 'c', 'd'], child)
		
	def test_pmx_child2(self):
		mother = ['a', 'c', 'd', 'b', 'e']
		father = ['c', 'a', 'b', 'e', 'd']
		
		child = oblig1.pmx_child(mother, father, 2, 4)
		self.assertEqual(['c', 'a', 'd', 'b', 'e'], child)
		
	def test_pmx_child4(self):
		mother = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
		father = ['9', '3', '7', '8', '2', '6', '5', '1', '4']
		
		child = oblig1.pmx_child(mother, father, 3, 6)
		self.assertEqual(['9', '3', '2', '4', '5', '6', '7', '1', '8'], child)

if __name__ == '__main__':
    unittest.main()