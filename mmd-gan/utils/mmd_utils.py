import torch
import random

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




def _bounded_rbf_kernel_repulsive(X, Y, sigma_list, upper_bound, lower_bound, lambda_repulsive):
    """
    PyTorch implementation of richardwth's TF implementation get_squared_dist() + bounded kernel funct
    https://github.com/richardwth/MMD-GAN/blob/master/GeneralTools/math_func.py
    
    dist_gg, dist_gd, dist_dd = get_squared_dist(
            self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
    """
    print("richardwth implementation")
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    xxt = torch.mm(X, X.t())  # [xi_xi, xi_xj; xj_xi, xj_xj], batch_size-by-batch_size
    xyt = torch.mm(X, Y.t())
    yyt = torch.mm(Y, Y.t())
    
    dx = torch.diag(xxt).unsqueeze(1)  # [xxt], [batch_size]
    dy = torch.diag(yyt).unsqueeze(1)
    
#     dist_xx = torch.max(dx.expand_as(xxt) - 2.0 * xxt + dx.expand_as(xxt), 0.0)
#     dist_xy = torch.max(dx.expand_as(xyt) - 2.0 * xyt + dy.expand_as(yyt), 0.0)
#     dist_yy = torch.max(dy.expand_as(yyt) - 2.0 * yyt + dy.expand_as(yyt), 0.0)
    dist_xx = dx.expand_as(xxt) - 2.0 * xxt + dx.expand_as(xxt)
    dist_xy = dx.expand_as(xyt) - 2.0 * xyt + dy.expand_as(yyt)
    dist_yy = dy.expand_as(yyt) - 2.0 * yyt + dy.expand_as(yyt)
    dist_xx.clamp_(min = 0.0)
    dist_xy.clamp_(min = 0.0)
    dist_yy.clamp_(min = 0.0)
#     print("sum of dist_xx: {}, dist_xy: {}, dist_yy: {}".format(sum(dist_xx),sum(dist_xy),sum(dist_yy)))
#     print("dist_xx (real,real) max: {}, min: {}".format(torch.max(dist_xx),torch.min(dist_xx)))
#     print("dist_xy (real,gen)  max: {}, min: {}".format(torch.max(dist_xy),torch.min(dist_xy)))
#     print("dist_yy (gen,gen)   max: {}, min: {}".format(torch.max(dist_yy),torch.min(dist_yy)))
    
    
    random_kernel_bandwidth = False
    if random_kernel_bandwidth:
        sigma = random.choice(sigma_list)
#         print("Randomly chosen kernel bandwidth: {}".format(sigma))
    else:
        sigma = sigma_list[0]
#         print("Nonrandom kernel bandwidth: {}".format(sigma))
    
    
    k_xx = torch.exp(-dist_xx / (2.0 * sigma ** 2))
    k_yy = torch.exp(-dist_yy / (2.0 * sigma ** 2))
    k_xy = torch.exp(-dist_xy / (2.0 * sigma ** 2))
    
#     custom_weights = [1.0,0.0]
    custom_weights = [-lambda_repulsive + 1, -lambda_repulsive]
#     print("custom_weights: {}".format(custom_weights))
    
    # in rep loss, custom_weights[0] - custom_weights[1] = 1
#     k_xx_b = torch.exp(-torch.max(dist_xx, lower_bound) / (2.0 * sigma ** 2))
    """
    BELOW CODE FROM RICHARDWTH, SWITCHED K_XX WITH K_YY BECAUSE K_YY IS REAL IN THE CODE
    """
#     print("lower bounding dist_yy (gen,gen) with {}".format(lower_bound))
#     print("dist_yy (gen,gen)   max: {}, min: {}".format(torch.max(dist_yy),torch.min(dist_yy)))
    dist_yy.clamp_(min = lower_bound)
#     print("dist_yy (gen,gen) lower bounded max: {}, min: {}".format(torch.max(dist_yy),torch.min(dist_yy)))
    k_yy_b = torch.exp(-dist_yy / (2.0 * sigma ** 2))
    if custom_weights[0] > 0:
#         print("upper bounding dist_xy (real,gen) with {} because custom_weight[0] = {} > 0".format(upper_bound,custom_weights[0]))
#         print("dist_xy (real,gen)  max: {}, min: {}".format(torch.max(dist_xy),torch.min(dist_xy)))
        dist_xy.clamp_(max = upper_bound)
#         print("dist_xy (real,gen) upper bounded max: {}, min: {}".format(torch.max(dist_xy),torch.min(dist_xy)))
#         k_xy_b = torch.exp(-torch.min(dist_xy, upper_bound) / (2.0 * sigma ** 2))
        k_xy_b = torch.exp(-dist_xy / (2.0 * sigma ** 2))
    else:
        k_xy_b = k_xy  # no lower bound should be enforced as k_xy may be zero at equilibrium
    if custom_weights[1] > 0:  # the original mmd-g
#         print("lower bounding dist_xx (real,real) with {} because custom_weight[1] = {} > 0".format(lower_bound,custom_weights[1]))
#         print("dist_xx (real,real) max: {}, min: {}".format(torch.max(dist_xx),torch.min(dist_xx)))
        dist_xx.clamp_(min = lower_bound)
#         print("dist_xx (real,real) lower bounded max: {}, min: {}".format(torch.max(dist_xx),torch.min(dist_xx)))
#         k_yy_b = torch.exp(-torch.max(dist_yy, lower_bound) / (2.0 * sigma ** 2))
        k_xx_b = torch.exp(-dist_xx / (2.0 * sigma ** 2))
    else:  # the repulsive mmd-g
#         print("upper bounding dist_xx (real,real) with {}".format(upper_bound))
#         k_yy_b = torch.exp(-torch.min(dist_yy, upper_bound) / (2.0 * sigma ** 2))
#         print("dist_xx (real,real) max: {}, min: {}".format(torch.max(dist_xx),torch.min(dist_xx)))
        dist_xx.clamp_(max = upper_bound)
#         print("dist_xx (real,real) upper bounded max: {}, min: {}".format(torch.max(dist_xx),torch.min(dist_xx)))
        k_xx_b = torch.exp(-dist_xx / (2.0 * sigma ** 2))
    
#     print("k_xx_b (real, real) max: {}, min: {}".format(torch.max(k_xx_b),torch.min(k_xx_b)))
#     print("k_xy_b (real, gen)  max: {}, min: {}".format(torch.max(k_xy_b),torch.min(k_xy_b)))
#     print("k_yy_b (gen, gen)   max: {}, min: {}".format(torch.max(k_yy_b),torch.min(k_yy_b)))
#     print("sizes of k_xx_b: {}, k_xy_b: {}, k_yy_b: {}".format(k_xx_b.size(),k_xy_b.size(),k_yy_b.size()))
    
#     return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)
#     print("K_XX: {}, K_XX_B: {}, K_XY: {}, K_XY_B: {}, K_YY: {}, K_YY_B: {}".format(k_xx,k_xx_b,k_xy,k_xy_b,k_yy,k_yy_b)) 

    return [k_xx_b, k_xy_b, k_yy_b], [k_xx, k_xy, k_yy], len(sigma_list)


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
#     print("exponent[:m, :m] (XX) max: {}, min: {}".format(torch.max(exponent[:m, :m]),torch.min(exponent[:m, :m])))

#     # K_XY
#     exponent[:m, m:].clamp_(max = upper_bound) # richardwth line 1388 ????? Not sure to include
#     exponent[:m, m:] = torch.min(exponent[:m, m:].clone(), 
#                           torch.ones_like(exponent[:m, m:]) * upper_bound)
#     exponent[:m, m:] = torch.max(exponent[:m, m:].clone(), 
#                           torch.ones_like(exponent[:m, m:]) * lower_bound)
#     print("exponent[:m, m:] (XY) max: {}, min: {}".format(torch.max(exponent[:m, m:]),torch.min(exponent[:m, m:])))
    
#     # K_YY
    exponent[m:, m:].clamp_(max = upper_bound) # richardwth line 1394
#     exponent[m:, m:].clamp_(min = lower_bound)
#     exponent[m:, m:] = torch.clamp(exponent[m:, m:], min = lower_bound)
#     exponent[m:, m:] = torch.max(exponent[m:, m:].clone(), 
#                           torch.ones_like(exponent[m:, m:]) * lower_bound)    
#     exponent[m:, m:] = torch.min(exponent[m:, m:].clone(), 
#                           torch.ones_like(exponent[m:, m:]) * upper_bound)    
#     print("exponent[m:, m:] (YY) max: {}, min: {}".format(torch.max(exponent[m:, m:]),torch.min(exponent[m:, m:])))


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
#     print("X = " + str(X))
#     print("Y = " + str(Y))
#     print("X size = " + str(X.size()))
#     print("Y size = " + str(Y.size()))
    
    assert(X.size(0) == Y.size(0))
    m = X.size(0) # m = batch size

    # X and Y is the f_enc_X_D -> dimensions are the output of encoder
    # Z is the two arrays side by side
    Z = torch.cat((X, Y), dim = 0)
#     print("Z = " + str(Z))
#     print("Z size = " + str(Z.size()))
    
    ZZT = torch.mm(Z, Z.t())
#     print("ZZT = " + str(ZZT))
#     print("ZZT size = " + str(ZZT.size()))
    
    # 
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
#     print("diag_ZZT = " + str(diag_ZZT))
#     print("diag_ZZT size = " + str(diag_ZZT.size()))
    
    # Expand this tensor to the same size as other
    # 
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
#     print("Z_norm_sqr = " + str(Z_norm_sqr))
#     print("Z_norm_sqr size = " + str(Z_norm_sqr.size()))
    
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t() # ||x-y||^2
#     print("exponent size = " + str(exponent.size()))
#     print("exponent = " + str(exponent))


    # line 805, 833 & 834 in richardwth TF implementation
    exponent.clamp_(min = 0.0)
    
#     print("exponent[:m, :m] (XX) max: {}, min: {}".format(torch.max(exponent[:m, :m]),torch.min(exponent[:m, :m])))
#     print("exponent[:m, m:] (XY) max: {}, min: {}".format(torch.max(exponent[:m, m:]),torch.min(exponent[:m, m:])))
#     print("exponent[m:, m:] (YY) max: {}, min: {}".format(torch.max(exponent[m:, m:]),torch.min(exponent[m:, m:])))



    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2.0 * sigma**2) # (1/(2sigma^2))
        K += torch.exp(-gamma * exponent) # Eq.5 in Demystifying MMD GANs
#         print("sigma = " + str(sigma))
#         print("gamma = " + str(gamma))
#         print("exponent = " + str(exponent))
#         print("added to K (=torch.exp(-gamma * exponent)) " + str(torch.exp(-gamma * exponent)))
#     print("K = " + str(K))
#     print("K size = " + str(K.size()))

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True, 
                 repulsive_loss = False, lambda_repulsive = False, 
                 bounded_kernel = False, upper_bound = False, lower_bound = False):
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
#     if bounded_kernel == True:
    if repulsive_loss == True:
#         upper_bound = 4.0
#         lower_bound = 0.25
#         K_XX, K_XY, K_YY, d = _bounded_rbf_kernel(X, Y, sigma_list, 
#                                                   upper_bound = upper_bound, 
#                                                   lower_bound = lower_bound)
#         print("_bounded_rbf_kernel: KXX {}, KXY, {}, KYY {}".format(sum(K_XX), sum(K_XY), sum(K_YY)))
#         print("_bounded_rbf_kernel: KXX {}, KXY, {}, KYY {}".format(K_XX, K_XY, K_YY))
#         print("_bounded_rbf_kernel_repulsive")
        K_list, _ , d = _bounded_rbf_kernel_repulsive(X, Y, 
                                                            sigma_list, 
                                                            upper_bound, 
                                                            lower_bound,
                                                            lambda_repulsive)
        K_XX, K_XY, K_YY = K_list[0],K_list[1],K_list[2]
#         print("_bounded_rbf_kernel_repulsive: KXX {}, KXY, {}, KYY {}".format(sum(K_XX), sum(K_XY), sum(K_YY)))
#         print("_bounded_rbf_kernel_repulsive: KXX {}, KXY, {}, KYY {}".format(K_XX, K_XY, K_YY))
#         print("Upper bound: {}, lower bound: {}".format(upper_bound,lower_bound))
    
#         print("_mmd2_repulsive")
        return _mmd2_repulsive(K_XX, K_XY, K_YY, const_diagonal=False, 
                 biased=False, bounded_kernel = bounded_kernel,
                 repulsive_loss = repulsive_loss, lambda_repulsive = lambda_repulsive)
        
    else:
#         K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
#         print("_bounded_rbf_kernel_repulsive but uses unbounded Ks")
        _ , K_list , d = _bounded_rbf_kernel_repulsive(X, Y, 
                                                            sigma_list, 
                                                            upper_bound, 
                                                            lower_bound,
                                                            lambda_repulsive)
        K_XX, K_XY, K_YY = K_list[0],K_list[1],K_list[2]

#     print("K_XX size = " + str(K_XX.size()))
#     print("K_XY size = " + str(K_XY.size()))
#     print("K_YY size = " + str(K_YY.size()))
    
#         print("_mmd2_new")
        return _mmd2_new(K_XX, K_XY, K_YY, const_diagonal=False, 
                     biased=biased, bounded_kernel = bounded_kernel,
                     repulsive_loss = repulsive_loss, lambda_repulsive = lambda_repulsive)
    
#     return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, 
#                  biased=biased, bounded_kernel = bounded_kernel,
#                  repulsive_loss = repulsive_loss, lambda_repulsive = lambda_repulsive)



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


def _mmd2_repulsive(K_XX, K_XY, K_YY, const_diagonal=False, 
                 biased=False, bounded_kernel = True,
                 repulsive_loss = True, lambda_repulsive = 1.0):
    """
    Taken from richardwth's implementation
    https://github.com/richardwth/MMD-GAN/blob/master/GeneralTools/math_func.py  line #2530
    def _repulsive_mmd_g_bounded_(self):
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
        self.loss_gen, self.loss_dis = mmd_g_bounded(
            dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0, lower_bound=0.25, upper_bound=4.0,
            name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
    
            # inside mmd_g_bounded
            e_kxx = matrix_mean_wo_diagonal(k_xx, m)
                e_kxy = matrix_mean_wo_diagonal(k_xy, m)
                e_kyy = matrix_mean_wo_diagonal(k_yy, m)
                e_kxx_b = matrix_mean_wo_diagonal(k_xx_b, m)
                e_kyy_b = matrix_mean_wo_diagonal(k_yy_b, m)
                e_kxy_b = matrix_mean_wo_diagonal(k_xy_b, m) if custom_weights[0] < 0 else e_kxy
    
    # k_xx_b, k_yy_b, k_xy_b size = [batch_size x batch_size]
         
         
    # line 1048   
    def matrix_mean_wo_diagonal(matrix, num_row, num_col=None, name='mu_wo_diag'):
        # This function calculates the mean of the matrix elements not in the diagonal
        with tf.name_scope(name):
            if num_col is None:
                mu = (tf.reduce_sum(matrix) - tf.reduce_sum(tf.matrix_diag_part(matrix))) / (num_row * (num_row - 1.0))
            else:
                mu = (tf.reduce_sum(matrix) - tf.reduce_sum(tf.matrix_diag_part(matrix))) \
                     / (num_row * num_col - tf.minimum(num_col, num_row))

            return mu
            
    """
    m = K_XX.size(0)
    
    custom_weights = [-lambda_repulsive + 1, -lambda_repulsive]
#     print("custom_weights: {}".format(custom_weights))
    assert custom_weights[0] - custom_weights[1] == 1.0, 'w[0]-w[1] must be 1'
    
    EKxx = (K_XX.sum() - torch.diag(K_XX).sum()) / (m * (m - 1))
    EKyy = (K_YY.sum() - torch.diag(K_YY).sum()) / (m * (m - 1))
    EKxy = (K_XY.sum() - torch.diag(K_XY).sum()) / (m * m)
    
    mmd2 = -custom_weights[1] * EKxx - EKyy + custom_weights[0] * EKxy
#     print("mmd2 = -{}*EKxx - EKyy + {}EKxy".format(custom_weights[1],custom_weights[0]))

#     print("EKxx (real,real): {}, EKxy (real,gen): {}, EKyy (gen,gen): {}".format(EKxx,EKxy,EKyy))
#     print("MMD2: {}".format(mmd2))
    return mmd2 , [EKxx, EKyy, EKxy]


def _mmd2_new(K_XX, K_XY, K_YY, const_diagonal=False, biased=False, bounded_kernel = True,
          repulsive_loss = False, lambda_repulsive = False):
    
    m = K_XX.size(0)
    
#     custom_weights = [-lambda_repulsive + 1, -lambda_repulsive]
#     print("custom_weights: {}".format(custom_weights))
#     assert custom_weights[0] - custom_weights[1] == 1.0, 'w[0]-w[1] must be 1'
    
    EKxx = (K_XX.sum() - torch.diag(K_XX).sum()) / (m * (m - 1))
    EKyy = (K_YY.sum() - torch.diag(K_YY).sum()) / (m * (m - 1))
    EKxy = (K_XY.sum() - torch.diag(K_XY).sum()) / (m * m)
    
    mmd2 = EKxx + EKyy -2*EKxy
    print("mmd2 = EKxx + EKyy -2*EKxy")

    print("EKxx (real,real): {}, EKxy (real,gen): {}, EKyy (gen,gen): {}".format(EKxx,EKxy,EKyy))
    print("MMD2: {}".format(mmd2))
    return mmd2 , [EKxx, EKyy, EKxy]    
    

# def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False, bounded_kernel = True,
          repulsive_loss = False, lambda_repulsive = False):
    """
    Inputs:
        K_XX = K[:m, :m] size = [batch_size, batch_size]
        K_XY = K[:m, m:] size = [batch_size, batch_size]
        K_YY = K[m:, m:] size = [batch_size, batch_size]
        
    m = batch_size

    """
#     print("K_XX: {}".format(K_XX))
#     print("K_YY: {}".format(K_YY))
#     print("K_XY: {}".format(K_XY))
    
    
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
#     print("K_XX.sum(dim=1): {}".format(K_XX.sum(dim=1)))
#     print("K_YY.sum(dim=1): {}".format(K_YY.sum(dim=1)))
#     print("K_XY.sum(dim=0): {}".format(K_XY.sum(dim=0)))
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
#     print("K_XY_sum: {}".format(K_XY_sum)) 

    # Biased Original MMD Loss
    if biased and not repulsive_loss:
        print("Original and Biased MMD")
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
        
        K_XX_contribution = - lambda_repulsive * (Kt_XX_sum + sum_diag_X) / (m * m)
        K_YY_contribution = (Kt_YY_sum + sum_diag_Y) / (m * m)
        K_XY_contribution = (lambda_repulsive - 1.0) * K_XY_sum / (m * m)
        
        mmd2 = K_XX_contribution + K_YY_contribution + K_XY_contribution
        
        print("K_XX: {}, K_XY: {}, K_YY: {}".format(K_XX_contribution,K_XY_contribution,K_YY_contribution))
        print("MMD2: {}".format(mmd2))
        
        # reverse the signs
#         mmd = -1.0 * mmd2 
#         mmd2 = -1.0 * ((lambda_rep * (Kt_XX_sum + sum_diag_X)) / (m * m) \
#                 - (lambda_rep - 1.0) * K_XY_sum / (m * m) \
#                 - (Kt_YY_sum + sum_diag_Y) / (m * m))
        return mmd2 , [K_XX_contribution, K_YY_contribution, K_XY_contribution]
#         return mmd2

    elif not biased and repulsive_loss:
        print("Repulsive and Unbiased MMD")
        
        # my implementation
        # Lambda_rep = 1 -> KXX - KYY
        # Lambda_rep = 0.5 -> 0.5KXX + 0.5KXY - KYY
        # Lambda_rep = 0 -> KXY - KYY
        # Lambda_rep = -1 -> 2KXY - KXX - KYY
        # should be opposite signed errD.backward(-1)
#         K_XX_contribution = -lambda_repulsive * Kt_XX_sum / (m * (m - 1)) 
#         K_YY_contribution = Kt_YY_sum / (m * (m - 1)) 
#         K_XY_contribution = (lambda_repulsive - 1.0) * K_XY_sum / (m * m) 
        
        # richardwth math_func.py line 1421 mmd2
        # k_yy = real data in richardwth (line #2115)
#         w_k_xy = 1.0
#         w_k_xx = 0.0
#         assert w_k_xy - w_k_xx == 1.0, 'w_k_xy-w_k_xx must be 1'
#         K_XX_contribution = -w_k_xx * Kt_XX_sum / (m * (m - 1)) 
#         K_YY_contribution = -Kt_YY_sum / (m * (m - 1)) 
#         K_XY_contribution = w_k_xy * K_XY_sum / (m * m) 


        # -c_w[1] = lambda_repulsive
        # c_w[0] = -lambda_repulsive + 1
#         custom_weights = [1.0,0.0]
        custom_weights = [-lambda_repulsive + 1, -lambda_repulsive]
        print("custom_weights: {}".format(custom_weights))
        assert custom_weights[0] - custom_weights[1] == 1.0, 'w[0]-w[1] must be 1'
        EKxx = Kt_XX_sum / (m * (m - 1)) 
        EKyy = Kt_YY_sum / (m * (m - 1)) 
        EKxy = K_XY_sum / (m * m) 
        
        mmd2 = -custom_weights[1] * EKxx - EKyy + custom_weights[0] * EKxy
        print("mmd2 = -{}*EKxx - EKyy + {}EKxy".format(custom_weights[1],custom_weights[0]))
        
        print("EKxx (real,real): {}, EKxy (real,gen): {}, EKyy (gen,gen): {}".format(EKxx,EKxy,EKyy))
        print("MMD2: {}".format(mmd2))
        return mmd2 , [EKxx, EKyy, EKxy]

    # Not biased Original MMD Loss
    else:
        print("Original and Unbiased MMD")
        # original implementation
        # L_D_att = 2KXY - KXX - KYY -> the original implementation is opposite signed
        # Generator uses this loss function -> Eq. 2 in Repulsive Loss paper
        K_XX_contribution = Kt_XX_sum / (m * (m - 1))
        K_YY_contribution = Kt_YY_sum / (m * (m - 1))
        K_XY_contribution = - 2.0 * K_XY_sum / (m * m)
        print("K_XX: {}, K_XY: {}, K_YY: {}".format(K_XX_contribution,K_XY_contribution,K_YY_contribution))
        
        mmd2 = K_XX_contribution + K_YY_contribution + K_XY_contribution
        print("MMD2: {}".format(mmd2))
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
        
        
        
