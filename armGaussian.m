classdef armGaussian<handle
    
    properties
        mean % Double : Expectations of the arm for each state
        v    % Double : Variances of the arm for each state
    end
    
    methods
        function self = armGaussian(m, v)
            self.mean = m;
            self.v = v;
        end
        
        function [reward] = sample(self)
            reward = self.mean + self.v*randn(1);
        end
                
    end
end
