#include<iostream>
#include<exception>
using namespace std;

class MyExc:public exception
{
	public:
	const char *what() const throw()
	{
		return "You have entered wrong password three times. Transaction terminated\n";
	}
};

class Account
{
	public:
	static double bal;
    void cb()
	{
		cout<<"Available balance in account is "<<bal<<endl;
	}
};

double Account::bal = 8000;

class Transactions:public Account
{
	private:
	int w,a;
	public:
	void wc()
	{
		cout<<"Enter the amount you wish to withdraw"<<endl;
		cin>>w;
		if(w%100!=0)
		{
			cout<<"\nPlease enter an amount in denomination of 100s"<<endl;
		}
		else if(w>bal)
		{
			cout<<"\nPlease enter an amount less than or equal to balance"<<endl;
		}
		else
		{
			bal = bal - w;
			cout<<"\nAmount withdrawn = "<<w<<endl;
			cb();
		}
	}
	
	void da()
	{
		cout<<"Enter the amount you wish to deposit"<<endl;
		cin>>a;
		if(a%100!=0)
		{
			cout<<"Please enter an amount in denomination of 100s"<<endl;
		}
		else
		{
			bal = bal + a;
			cout<<"\nDeposit Amount = "<<a<<endl;
			cb();
		}
	}
};

int main()
{
	
	int f=0,p,choice,password = 8965;
	
	cout<<"Please enter your password"<<endl;
	cin>>p;
	
	while(p!=password)
	{
		cout<<"Entered password is incorrect. Please try again\n";
		cin>>p;
		f++;
		try{
		if(f==3)
		{
			MyExc m;
			throw m;
		}
		}catch (exception &e)
		{
			cout<<e.what();
			exit(0);
		}
	}
	
	Transactions t;

	do
	{	
	cout<<"\nWelcome to ATM services"<<endl<<endl;
	cout<<"Choose an option to continue"<<endl;
	cout<<"1.Check current balance\n";
	cout<<"2.Withdraw cash\n";
	cout<<"3.Deposit amount\n";
	cout<<"4.Quit\n";
	
	cin>>choice;
	
	switch(choice)
	{
		case 1: cout<<"\nYour current balance in account is "<<Account::bal<<endl;
				break;
				
		case 2: t.wc();
				break;
				
		case 3: t.da();
				break;
				
		case 4: exit(0);
		
		default: cout<<"\nPlease enter the correct denoted choice"<<endl;
					  break;
	}
	}while(choice!=4);
}
