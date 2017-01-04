classdef armGaussian<handle
	
	properties
		h	 % Int : Which state is hot
		mean % (2*N) : Expectations of each arm for each state
		var	 % (2*N) : Variances of each arm for each state
		p	 % Double: Probability to stay in normal state
		ptoH % (1*N) : Probabilities that one arm became hot
		ptoN % (1*N) : Probabilities that each hot arm go back to normal
	end
	
	methods
		function self = armGaussian(mean, var, ptoH, ptoN)
			self.h=0; 
			self.mean = mean;
			self.var = var;
			self.p = 1-sum(ptoH);
			self.ptoH = ptoH/sum(ptoH);
			self.ptoN = ptoN;
		end
		
		function [reward] = sample(self, s)
			reward = self.mean(1+(h==s),s) + self.var(1+(h==s),s)*randn(1);
			if (h == 0)
				if (rand(1) > self.p) 
					[~,h] = max(mnrnd(1,ptoH));
				end
			else
				h = h*(rand(1)>self.ptoN(1,h));
			end
		end
				
	end	   
end
