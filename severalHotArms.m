classdef severalHotArms<handle
    
    properties
        h    % (S*N) : Which state is hot
        %mean % (S*N) : Expectations of each arm for each state
        %v    % (2*N) : Variances of each arm for each state
        %p    % Double: Probability to stay in normal state
        p    % (S*N) : Transition probabilities
        arms % {S*N} : Each arm
    end
    
    methods
        function self = severalHotArms(istate, type, mean, v, p)
            self.h=istate;
            for s=1:S
                for n=1:N
                    switch type{s,n}
                        case 'gaussian'
                            self.arms{s,n} = armGaussian(mean(s,n), v(s, n));
                    end
                end
            end
            self.p = p;
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
