classdef severalArmGaussian<handle
	
	properties
		h	 % Int : Which state is hot
		mean % (2*N) : Expectations of each arm for each state
		v	 % (2*N) : Variances of each arm for each state
		p	 % Double: Probability to stay in normal state
		ptoH % (1*N) : Probabilities that one arm became hot
		ptoN % (1*N) : Probabilities that each hot arm go back to normal
	end
	
	methods
		function self = severalArmGaussian(mean, v, ptoH, ptoN)
			self.h=0; 
			self.mean = mean;
			self.v = v;
			self.p = 1-sum(ptoH);
			self.ptoH = ptoH/sum(ptoH);
			self.ptoN = ptoN;
		end
		
		function [reward] = sample(self, s)
			reward = self.mean(1+(self.h==s),s) + self.v(1+(self.h==s),s)*randn(1);
			if (self.h == 0)
				if (rand(1) > self.p) 
					[~,self.h] = max(mnrnd(1,self.ptoH));
				end
			else
				self.h = self.h*(rand(1)>self.ptoN(1,self.h));
			end
		end
				
	end
end
