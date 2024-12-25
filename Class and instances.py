import datetime

class Employee:

	num_of_emp = 0
	raise_amount = 1.05
	
	def __init__ (self, first, last, pay):
		self.first = first
		self.last = last
		self.pay = pay
		self.email = first +'.'+last +"@gmail.com"
		Employee.num_of_emp +=1


	def fullname(self):
		return '{} {}'.format(self.first, self.last)	
	
	def apply_raise(self):
		self.pay = int(self.pay * self.raise_amount)   #we can also use Emplyoee.raise_amount
#REGULAR method above below we use class method

	@classmethod
	def set_raise_amt(cls, amount):
		cls.raise_amt = amount

	@classmethod
	def from_string(cls, emp_string):
		first, last, pay = emp_string.split('-')
		return cls(first, last, pay)
	
	@staticmethod
	def is_workday(day):
		return day.weekday() not in (4,5)
	
class Developer(Employee):

	raise_amt = 1.10
	def __init__ (self, first, last, pay,prog_lang):
		super().__init__(first, last, pay)
		#Employee.__init__(self, first, last, pay) U can use this but super is better for multiple inheritance and sustainable
		self.prog_lang = prog_lang

class Manager(Employee):
	def __init__ (self, first, last, pay,employees = None):
		super().__init__(first, last, pay)
		if employees is None:
			self.employees = []
		else:
			self.employees = employees
	
	def add_emp(self,emp):
		if emp not in self.employees:
			self.employees.append(emp)

	def remove_emp(self,emp):
		if emp in self.employees:
			self.employees.remove(emp)
	
	def print_emps(self):
		for emp in self.employees:
			print('-->', emp.fullname())

emp_1 = Employee('Pratham','Bhandari',50000)
emp_2 = Employee("test","User",40000)


dev_1 = Developer("lewis",'Hamilton',40000,'python')
dev_2 = Developer("warrs",'asdwa',30000,'Java')



mgr_1 = Manager('leo','Messi',90000,[dev_1])

#print(mgr_1.email)

mgr_1.add_emp(dev_2)
mgr_1.remove_emp(dev_1)

mgr_1.print_emps()
#print(dev_1.email)

#my_date = datetime.date(2024,12,11)
#print(Employee.is_workday(my_date))

'''
emp_str_1 = 'jhon-doe-70000'
emp_str_2 = 'steve-smith-80000'
emp_str_3 = 'hermit-crab-90000'

	new_emp_1 = Employee.from_string(emp_str_1)
	new_emp_2 = Employee.from_string(emp_str_2)
'''

Employee.set_raise_amt(1.07)

#print(emp_1.fullname())
#print(emp_1.email)
#print(f"Name: {emp_1.fullname()} Email: {emp_1.email} Payment: {emp_1.pay}")

#print(emp_1.raise_amount)
#print(Employee.raise_amount)

#print(emp_1.__dict__)

#print(Employee.num_of_emp)
#print(Employee.raise_amount)
#print(emp_1.raise_amt)
#print(emp_2.raise_amt)

#print(new_emp_2.email)

