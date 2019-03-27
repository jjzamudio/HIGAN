import torch

# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss

# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean

def _bounded_rbf_kernel(X, Y, sigma_list, upper_bound, lower_bound):
    """
    Proposed in Improving MMD GAN Training with Repulsive Loss Function
    Equation 6 & 7.
    
    upper_bound
    lower_bound
    
    TF implementation
    https://github.com/richardwth/MMD-GAN/blob/master/GeneralTools/math_func.py
    """
#     print("Bounded RBF Kernel")
    
#     print("X: {}".format(X))
#     print("Y: {}".format(Y))
    
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    # X and Y is the f_enc_X_D -> dimensions are the output of encoder
    # Z is the two arrays side by side
    Z = torch.cat((X, Y), dim = 0)
#     print("Z size = " + str(Z.size()))
#     print("Z: {}".format(Z))
    
    ZZT = torch.mm(Z, Z.t())
#     print("ZZT: {}".format(ZZT))
#     print("ZZT size = " + str(ZZT.size()))
    
    # 
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
#     print("diag_ZZT: {}".format(diag_ZZT))
#     print("diag_ZZT size = " + str(diag_ZZT.size()))
    
    # Expand this tensor to the same size as other
    # 
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
#     print("Z_norm_sqr: {}".format(Z_norm_sqr))
#     print("Z_norm_sqr size = " + str(Z_norm_sqr.size()))
    
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t() # ||x-y||^2
#     print("exponent size = " + str(exponent.size()))
#     print("exponent = " + str(exponent))

    
    # line 805, 833 & 834 in richardwth TF implementation
    exponent.clamp_(min = 0.0)
    
    

    # max(||a-b||^2, b_l) & min(||a-b||^2, b_u)
#     K_XX
    exponent[:m, :m].clamp_(min = lower_bound) # richardwth line 1386
#     exponent[:m, :m].clamp_(max = upper_bound)
#     exponent[:m, :m] = torch.clamp(exponent[:m, :m], min = lower_bound)
#     exponent[:m, :m] = torch.max(exponent[:m, :m].clone(), 
#                           torch.ones_like(exponent[:m, :m]) * lower_bound)
#     exponent[:m, :m] = torch.min(exponent[:m, :m].clone(), 
#                           torch.ones_like(exponent[:m, :m]) * upper_bound)
    print("exponent[:m, :m] (XX) max: {}, min: {}".format(torch.max(exponent[:m, :m]),torch.min(exponent[:m, :m])))

#     # K_XY
    exponent[:m, m:].clamp_(max = upper_bound) # richardwth line 1388 ????? Not sure to include
#     exponent[:m, m:] = torch.min(exponent[:m, m:].clone(), 
#                           torch.ones_like(exponent[:m, m:]) * upper_bound)
#     exponent[:m, m:] = torch.max(exponent[:m, m:].clone(), 
#                           torch.ones_like(exponent[:m, m:]) * lower_bound)
    print("exponent[:m, m:] (XY) max: {}, min: {}".format(torch.max(exponent[:m, m:]),torch.min(exponent[:m, m:])))
    
#     # K_YY
    exponent[m:, m:].clamp_(max = upper_bound) # richardwth line 1394
#     exponent[m:, m:].clamp_(min = lower_bound)
#     exponent[m:, m:] = torch.clamp(exponent[m:, m:], min = lower_bound)
#     exponent[m:, m:] = torch.max(exponent[m:, m:].clone(), 
#                           torch.ones_like(exponent[m:, m:]) * lower_bound)    
#     exponent[m:, m:] = torch.min(exponent[m:, m:].clone(), 
#                           torch.ones_like(exponent[m:, m:]) * upper_bound)    
    print("exponent[m:, m:] (YY) max: {}, min: {}".format(torch.max(exponent[m:, m:]),torch.min(exponent[m:, m:])))


    K = 0.0
    for sigma in sigma_list:
        
        gamma = 1.0 / (2.0 * sigma**2)    # (1/(2sigma^2))
        K += torch.exp(-gamma * exponent) # Eq.5 in Demystifying MMD GANs
        
#         print("sigma = " + str(sigma))
#         print("gamma = " + str(gamma))
#         print("exponent = " + str(exponent))
#         print("added to K (=torch.exp(-gamma * exponent)) " + str(torch.exp(-gamma * exponent)))
#     print("K size = " + str(K.size()))

    """
    K[:m, :m] =  K_XX -> lower_bounded
    K[:m, m:] =  K_XY -> upper_bounded
    K[m:, m:] =  K_YY -> lower_bounded
    """ 
#     print("K[:m, :m] =  K_XX: {}".format(K[:m, :m]))
#     print("K[:m, m:] =  K_XY: {}".format(K[:m, m:]))
#     print("K[m:, m:] =  K_YY: {}".format(K[m:, m:]))
    
    # Bounding the Kernels
#     # K_XX
#     K[:m, :m] = torch.max(K[:m, :m].clone(), 
#                           torch.ones_like(K[:m, :m]) * lower_bound)
#     # K_XY
#     K[:m, m:] = torch.min(K[:m, m:].clone(), 
#                           torch.ones_like(K[:m, m:]) * upper_bound)
#     # K_YY
#     K[m:, m:] = torch.max(K[m:, m:].clone(), 
#                           torch.ones_like(K[m:, m:]) * lower_bound)
    
    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)



def _mix_rational_quadratic_kernel(X, Y, alpha_list):
    """
    Demystifying MMD GANs suggests this kernel 
    k^rq_alpha(x,y) = (1 + ||x-y||^2/(2*alpha))^-alpha
    """
    
    assert(X.size(0) == Y.size(0))
    m = X.size(0) # m = batch size
    
    Z = torch.cat((X, Y), dim = 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    
    # ||x-y||^2
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t() 

    K = 0.0
    for alpha in alpha_list:
        
        rq_kernel = (1.0 + exponent/(2*alpha))**(-alpha)
        K += rq_kernel # Eq.6 in Demystifying MMD GANs
#         print("sigma = " + str(sigma))
#         print("gamma = " + str(gamma))
#         print("exponent = " + str(exponent))
#         print("added to K (=torch.exp(-gamma * exponent)) " + str(torch.exp(-gamma * exponent)))
#     print("K size = " + str(K.size()))

    return K[:m, :m], K[:m, m:], K[m:, m:], len(alpha_list)



def _mix_rbf_kernel(X, Y, sigma_list):
    """
    Inputs:
        X -> f_enc_X_D ->
            size = batch_size x nz 
                nz = hidden dimension of z
        Y -> f_enc_Y_D -> 
            size = batch_size x nz 
                nz = hidden dimension of z
        sigma_list -> 
            base = 1.0
            sigma_list = [1, 2, 4, 8, 16]
            sigma_list = [sigma / base for sigma in sigma_list] 
            
    m = batch_size
    torch.cat(seq, dim=0, out=None) → Tensor
        Concatenates the given sequence of seq tensors 
        in the given dimension
    Z size = [2 x batch_size, nz]
    
    torch.mm(mat1, mat2, out=None) → Tensor
        Performs a matrix multiplication of the matrices mat1 and mat2
    ZZT size = [2 x batch_size, 2 x batch_size]
    
    torch.diag(input, diagonal=0, out=None) → Tensor
        If input is a matrix (2-D tensor), then returns a 1-D tensor 
        with the diagonal elements of input
    torch.unsqueeze(input, dim, out=None) → Tensor
        Returns a new tensor with a dimension of size 
        one inserted at the specified position
    diag_ZZT = [2 x batch_size, 1]
    
    expand_as(other) → Tensor
        Expand this tensor to the same size as other
    Z_norm_sqr = [2 x batch_size, 2 x batch_size]
    
    torch.exp(tensor, out=None) → Tensor
        Returns a new tensor with the exponential of the elements of input
        y_i = e^(x_i)
        
    exponent size = [2 x batch_size, 2 x batch_size]
    K size = [2 x batch_size, 2 x batch_size]
    """
#     print("X size = " + str(X.size()))
#     print("Y size = " + str(Y.size()))
    
    assert(X.size(0) == Y.size(0))
    m = X.size(0) # m = batch size

    # X and Y is the f_enc_X_D -> dimensions are the output of encoder
    # Z is the two arrays side by side
    Z = torch.cat((X, Y), dim = 0)
#     print("Z size = " + str(Z.size()))
    
    ZZT = torch.mm(Z, Z.t())
#     print("ZZT size = " + str(ZZT.size()))
    
    # 
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
#     print("diag_ZZT size = " + str(diag_ZZT.size()))
    
    # Expand this tensor to the same size as other
    # 
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
#     print("Z_norm_sqr size = " + str(Z_norm_sqr.size()))
    
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t() # ||x-y||^2
#     print("exponent size = " + str(exponent.size()))
#     print("exponent = " + str(exponent))


    # line 805, 833 & 834 in richardwth TF implementation
    exponent.clamp_(min = 0.0)
    
    print("exponent[:m, :m] (XX) max: {}, min: {}".format(torch.max(exponent[:m, :m]),torch.min(exponent[:m, :m])))
    print("exponent[:m, m:] (XY) max: {}, min: {}".format(torch.max(exponent[:m, m:]),torch.min(exponent[:m, m:])))
    print("exponent[m:, m:] (YY) max: {}, min: {}".format(torch.max(exponent[m:, m:]),torch.min(exponent[m:, m:])))



    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2.0 * sigma**2) # (1/(2sigma^2))
        K += torch.exp(-gamma * exponent) # Eq.5 in Demystifying MMD GANs
#         print("sigma = " + str(sigma))
#         print("gamma = " + str(gamma))
#         print("exponent = " + str(exponent))
#         print("added to K (=torch.exp(-gamma * exponent)) " + str(torch.exp(-gamma * exponent)))
#     print("K size = " + str(K.size()))

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True, 
                 repulsive_loss = False, bounded_kernel = False):
# def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    """
    How it is used in the training loop:
        mmd2_D = mix_rbf_mmd2(f_enc_X_D,  f_enc_Y_D,  sigma_list)
        X -> f_enc_X_D ->j
            size = batch_size x nz 
                nz = hidden dimension of z
        Y -> f_enc_Y_D -> 
            size = batch_size x nz 
                nz = hidden dimension of z
        sigma_list -> 
            base = 1.0
            sigma_list = [1, 2, 4, 8, 16]
            sigma_list = [sigma / base for sigma in sigma_list]
        
    _mix_rbf_kernel's internal K has [2 x batch_size, 2 x batch_size] size
    K_XX = K[:m, :m] (left upper quadrant) -> size = [batch_size, batch_size]
    K_XY = K[:m, m:] (right upper and left lower quadrant) -> size = [batch_size, batch_size]
    K_YY = K[m:, m:] (right lower quadrant) -> size = [batch_size, batch_size]
    d = len(sigma_list)
        
    """
    if bounded_kernel == True:
        upper_bound = 4.0
        lower_bound = 0.25
        K_XX, K_XY, K_YY, d = _bounded_rbf_kernel(X, Y, sigma_list, 
                                                  upper_bound = upper_bound, 
                                                  lower_bound = lower_bound)
        print("Upper bound: {}, lower bound: {}".format(upper_bound,lower_bound))
        
    else:
        K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
#     print("K_XX size = " + str(K_XX.size()))
#     print("K_XY size = " + str(K_XY.size()))
#     print("K_YY size = " + str(K_YY.size()))
    
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, 
                 biased=biased, bounded_kernel = bounded_kernel,
                 repulsive_loss = repulsive_loss)



def mix_rq_mmd2(X, Y, alpha_list, biased=True, repulsive_loss = False):

    K_XX, K_XY, K_YY, d = _mix_rational_quadratic_kernel(X, Y, alpha_list)
#     print("K_XX size = " + str(K_XX.size()))
#     print("K_XY size = " + str(K_XY.size()))
#     print("K_YY size = " + str(K_YY.size()))

    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased, 
                 repulsive_loss = repulsive_loss)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True, min_var_est=1e-8):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, 
                           const_diagonal=False, 
                           biased=biased, 
                           min_var_est = min_var_est)


# def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False, bounded_kernel = True,
          repulsive_loss = False):
    """
    Inputs:
        K_XX = K[:m, :m] size = [batch_size, batch_size]
        K_XY = K[:m, m:] size = [batch_size, batch_size]
        K_YY = K[m:, m:] size = [batch_size, batch_size]
        
    m = batch_size

    """
#     print("K_XX: {}".format(K_XX))
#     print("K_YY: {}".format(K_YY))
    
    
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        
        diag_X = torch.diag(K_XX)                       # (m,) 
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
#         print("diag_X: {}".format(diag_X))
#         print("diag_Y: {}".format(diag_Y))
#         print("sum_diag_X: {}".format(sum_diag_X))
#         print("sum_diag_Y: {}".format(sum_diag_Y))

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e
#     print("Kt_XX_sums: {}".format(Kt_XX_sums))
#     print("Kt_YY_sums: {}".format(Kt_YY_sums))  

    # repulsive loss
#     if bounded_kernel == True:
#         upper_bound = 4.0
#         lower_bound = 0.25
#         print("Upper bound: {}, lower bound: {}".format(upper_bound,lower_bound))
#         Kt_XX_sums.clamp_(max = upper_bound)
#         Kt_YY_sums.clamp_(min = lower_bound)
#         print("Kt_XX_sums max: {}, min: {}".format(torch.max(Kt_XX_sums),torch.min(Kt_XX_sums)))
#         print("Kt_YY_sums max: {}, min: {}".format(torch.max(Kt_YY_sums),torch.min(Kt_YY_sums)))

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e
#     print("Kt_XX_sum: {}".format(Kt_XX_sum))
#     print("Kt_YY_sum: {}".format(Kt_YY_sum))   

    # Biased Original MMD Loss
    if biased and not repulsive_loss:
#     if biased:
        K_XX_contribution = (Kt_XX_sum + sum_diag_X) / (m * m)
        K_YY_contribution = (Kt_YY_sum + sum_diag_Y) / (m * m)
        K_XY_contribution = - 2.0 * K_XY_sum / (m * m)
        mmd2 = K_XX_contribution + K_YY_contribution + K_XY_contribution
        return mmd2 , [K_XX_contribution, K_YY_contribution, K_XY_contribution]
#         return mmd2

    # Repulsive Loss Function
    elif biased and repulsive_loss:
        print("Repulsive and Biased MMD")
        lambda_rep = repulsive_loss
        
        K_XX_contribution = - lambda_rep * (Kt_XX_sum + sum_diag_X) / (m * m)
        K_YY_contribution = (Kt_YY_sum + sum_diag_Y) / (m * m)
        K_XY_contribution = (lambda_rep - 1.0) * K_XY_sum / (m * m)
        
        mmd2 = K_XX_contribution + K_YY_contribution + K_XY_contribution
        
        print("K_XX: {}, K_XY: {}, K_YY: {}".format(K_XX_contribution,K_XY_contribution,K_YY_contribution))
        
        # reverse the signs
#         mmd = -1.0 * mmd2 
#         mmd2 = -1.0 * ((lambda_rep * (Kt_XX_sum + sum_diag_X)) / (m * m) \
#                 - (lambda_rep - 1.0) * K_XY_sum / (m * m) \
#                 - (Kt_YY_sum + sum_diag_Y) / (m * m))
        return mmd2 , [K_XX_contribution, K_YY_contribution, K_XY_contribution]
#         return mmd2

    elif not biased and repulsive_loss:
        print("Repulsive and Unbiased MMD")
        lambda_rep = repulsive_loss
        
        # my implementation
#         K_XX_contribution = -lambda_rep * Kt_XX_sum / (m * (m - 1)) 
#         K_YY_contribution = Kt_YY_sum / (m * (m - 1)) 
#         K_XY_contribution = (lambda_rep - 1.0) * K_XY_sum / (m * m) 
        
        # richardwth math_func.py line 1421 mmd2
        K_XX_contribution = -Kt_XX_sum / (m * (m - 1)) 
        K_YY_contribution = lambda_rep * Kt_YY_sum / (m * (m - 1)) 
        K_XY_contribution = -(lambda_rep - 1.0) * K_XY_sum / (m * m) 
        
        mmd2 = K_XX_contribution + K_YY_contribution + K_XY_contribution
        
        print("K_XX: {}, K_XY: {}, K_YY: {}".format(K_XX_contribution,K_XY_contribution,K_YY_contribution))
        
        # reverse the signs
#         mmd = -1.0 * mmd2 
#         mmd2 = -1.0 * ((lambda_rep * (Kt_XX_sum + sum_diag_X)) / (m * m) \
#                 - (lambda_rep - 1.0) * K_XY_sum / (m * m) \
#                 - (Kt_YY_sum + sum_diag_Y) / (m * m))
        return mmd2 , [K_XX_contribution, K_YY_contribution, K_XY_contribution]

    # Not biased Original MMD Loss
    else:
        print("Original and Unbiased MMD")
        # original implementation
        K_XX_contribution = Kt_XX_sum / (m * (m - 1))
        K_YY_contribution = Kt_YY_sum / (m * (m - 1))
        K_XY_contribution = - 2.0 * K_XY_sum / (m * m)
        print("K_XX: {}, K_XY: {}, K_YY: {}".format(K_XX_contribution,K_XY_contribution,K_YY_contribution))
        
        mmd2 = K_XX_contribution + K_YY_contribution + K_XY_contribution
#         mmd2 = (Kt_XX_sum / (m * (m - 1))
#             + Kt_YY_sum / (m * (m - 1))
#             - 2.0 * K_XY_sum / (m * m))
        return mmd2 , [K_XX_contribution, K_YY_contribution, K_XY_contribution]
#         return mmd2

    



def _mmd2_and_ratio(K_XX, K_XY, K_YY, 
                    const_diagonal=False, 
                    biased=False,
                    min_var_est= 1e-8):
    
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, 
                                       const_diagonal = const_diagonal, 
                                       biased = biased)
    
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, 
                                         min = min_var_est))
    
    return loss, mmd2, var_est



def _mmd2_and_variance(K_XX, K_XY, K_YY, 
                       const_diagonal=False, 
                       biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X     # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y     # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)             # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)             # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X      # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y      # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum  = (K_XY ** 2).sum()                    # \| K_{XY} \|_F^2

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    var_est = (
        2.0 / (m**2 * (m - 1.0)**2) * (2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4.0*m - 6.0) / (m**3 * (m - 1.0)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0*(m - 2.0) / (m**3 * (m - 1.0)**2) * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0*(m - 3.0) / (m**3 * (m - 1.0)**2) * (K_XY_2_sum) - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0 / (m**3 * (m - 1.0)) * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
        )
    return mmd2, var_est



def normalize(x, dim=1):
    """
    used only in match() when dist == cos
    """
    return x.div(x.norm(2, dim=dim).expand_as(x))


def match(x, y, dist):
    '''
    Computes distance between corresponding points points in `x` and `y`
    using distance `dist`.
    
    # compute L2-loss of AE
    L2_AE_X_D = match(x.view(batch_size, -1), f_dec_X_D, 'L2')
    L2_AE_Y_D = match(y.view(batch_size, -1), f_dec_Y_D, 'L2')
    '''
    
    if dist == 'L2':
        return (x - y).pow(2).mean()
    elif dist == 'L1':
        return (x - y).abs().mean()
    elif dist == 'cos':
        x_n = normalize(x)
        y_n = normalize(y)
        return 2 - (x_n).mul(y_n).mean()
    else:
        assert dist == 'none', 'wtf ?'
        
        
        
