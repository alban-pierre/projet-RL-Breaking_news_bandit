classdef armGaussian<handle
    
    properties
        s    % Int : Current state vector
        mean % (1*N) : Expectations of the arm for each state
        v    % (1*N) : Variances of the arm for each state
        A    % (N*N) : Transition matrix
    end
    
    methods
        function self = armGaussian(s, mean, v, A)
            self.s=s; 
            self.mean = mean;
            self.v = v;
            self.A = A./sum(A,2);
        end
        
        function [reward] = sample(self)
            reward = self.mean(1,self.s) + self.v(1,self.s)*randn(1);
            [~,self.s] = max(mnrnd(1,self.A(self.s,:)),[],2);
        end
                
    end
end
