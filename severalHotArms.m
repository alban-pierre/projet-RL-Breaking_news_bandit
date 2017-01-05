classdef severalHotArms<handle
    
    properties
        h    % {1*N}(1*Sn) : Which state is hot
        p    % {S*N}(1*Sn) : Transition probabilities
        arms % {S*N} : Each arm
    end
    
    methods
        function self = severalHotArms(istate, type, mean, v, p)
            self.h=istate;
            for n=1:size(type,1)
                for s=1:size(type,2)
                    switch type{s,n}
                        case 'gaussian'
                            self.arms{s,n} = armGaussian(mean{s,n}, v{s, n});
                        otherwise
                            self.arms{s,n}.sample = @() assert(false);
                    end
                end
            end
            self.p = p;
        end
        
        function [reward] = sample(self, s)
            reward = self.arms{self.h(1,s), s}.sample();
            for n=1:size(self.h,2)
                self.h(1,n) = mnrnd(1,self.p{self.h(1,n),n}) * (1:size(self.h{1,n}))');
            end
        end
               
    end
end
