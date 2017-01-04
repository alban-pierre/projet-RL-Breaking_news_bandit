classdef severalArmGaussian<handle
	
	properties
		s	 % Int : Current state vector (example : [0;1])
		mean % (1*N) : Expectations of the arm for each state
		var	 % (1*N) : Variances of the arm for each state
		A	 % (N*N) : Transition matrix
	end
	
	methods
		function self = armGaussian(s, mean, var, A)
			self.s=s; 
			self.mean = mean;
			self.var = var;
			self.A = A;
		end
		
		function [reward] = sample(self)
			reward = self.mean(1,self.s) + self.var(1,self.s)*randn(1);
			[~,self.s] = max(mnrnd(1,A(self.s,:)));
		end
				
	end	   
end
