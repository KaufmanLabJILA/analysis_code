from .imports import *

from .expfile import *
from .analysis import *
from .mathutil import *
# from .plotutil import *
from .imagutil import *
# from .mako import *
# from .adam import *
# plots the rois on an image
#
def plotrois(img, rois, bg_offset = 0, bgs = False):
    """Plots rois as boxes over image"""
    plt.imshow(img)
    ax1 = plt.gca()
    for roi in rois:
        ax1.add_patch(
        patches.Rectangle((roi[0], roi[2]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'black'))
        if bgs:
            ax1.add_patch(
            patches.Rectangle((roi[0]+bg_offset[0], roi[2]+bg_offset[1]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'red'))
    plt.show()
    
def plotFill(parsed_data, plotROIs = False):
    """Plots fill fration, both for first image in each experiment, and for each individual ROI"""
    if parsed_data.keySort.ndim==1:
        plt.errorbar(parsed_data.keySort, parsed_data.fill_first, yerr = parsed_data.error_first, fmt='k-', zorder = 1)
        plt.errorbar(parsed_data.keySort, parsed_data.fillfracs, yerr = parsed_data.error, fmt='-', color="0.5", zorder = 0)
        plt.errorbar(parsed_data.keySort, parsed_data.fill_last, yerr = parsed_data.error_last, fmt='-', color="0.75", zorder = 0)
        plt.ylabel('Fill Fraction', fontsize = 14)
        plt.xlabel(parsed_data.keyName, fontsize = 14)
        plt.title("File " + str(parsed_data.fnum))
        plt.legend(['1st image', 'Average', 'Last'])
        plt.show()
#             plt.plot(parsed_data.keySort, parsed_data.fill_first, 'k-', zorder = 1)
#             plt.plot(parsed_data.keySort, parsed_data.fill_last, 'r-', zorder = 0)
#             plt.ylabel('Fill Fraction', fontsize = 14)
#             plt.xlabel(parsed_data.keyName, fontsize = 14)
#             plt.title("File " + str(parsed_data.fnum))
#             plt.legend(['1st image', 'Average', 'Last'])
#             plt.show()
        print('average fill fraction: ' + str(np.mean(parsed_data.fill_first)))
        z = 1.96 # corresponds to 95% CI
        plt.show()
#             plt.plot(parsed_data.keySort, np.mean(parsed_data.roisums, axis = 2)[:,0,0], 'ko')
#             plt.show()

        if plotROIs:
            for i in range(len(rois)):
                fill_roi = parsed_data.fill_rois[:,i]
                error_roi = z*np.sqrt(fill_roi*(1-fill_roi)/parsed_data.points)
                plt.errorbar(parsed_data.keySort, fill_roi, yerr = error_roi, fmt='-', zorder = 1)
            plt.ylabel('Fill Fraction', fontsize = 14)
            plt.xlabel(parsed_data.keyName, fontsize = 14)
            plt.title("File " + str(parsed_data.fnum))
            plt.legend(range(len(parsed_data.rois)))
            plt.show()
    elif parsed_data.keySort.ndim==2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(parsed_data.keySort[:,0], parsed_data.keySort[:,1], parsed_data.fill_first, c = 'b', marker='o')
        ax.set_xlabel(parsed_data.keyName[0])
        ax.set_ylabel(parsed_data.keyName[1])
        ax.set_zlabel('Fill Fraction')
        ax.set_title("File " + str(parsed_data.fnum))
        plt.show()

def plotLosses(parsed_data):
    if parsed_data.keySort.ndim==1:
        ks, losses, losserrs = getLossData(parsed_data)
        plt.errorbar(parsed_data.keySort, losses, yerr=losserrs, fmt = 'r-')
        plt.plot(parsed_data.keySort, losses, 'k.')
        plt.xlabel(parsed_data.keyName)
        plt.ylabel('losses')
        plt.title("File " + str(parsed_data.fnum))
        plt.axis([min(parsed_data.keySort), max(parsed_data.keySort), 0, 100])
        plt.show()
    elif parsed_data.keySort.ndim==2:
        ks, losses, losserrs = getLossData(parsed_data)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
#             ax = fig.add_subplot(111)
        ax.scatter(parsed_data.keySort[:,0], parsed_data.keySort[:,1], losses, c = 'b', marker='o')
        ax.set_xlabel(parsed_data.keyName[0])
        ax.set_ylabel(parsed_data.keyName[1])
        ax.set_zlabel('Fill Fraction')
        ax.set_title("File " + str(parsed_data.fnum))
        plt.show()
    return ks, losses


def plotTimeFill(parsed_data, plotROIs = False):
    """Plots fill fration, both for first image in each experiment, and for each individual ROI"""

    if parsed_data.roisums.shape[0]!=1:
        raise ValueError('Cannot plot fill per cycle with variations.')
    roisums = parsed_data.roisums[0].reshape(parsed_data.nreps, parsed_data.npics//2, 2, len(parsed_data.rois))
    roisums = roisums.swapaxes(0,1)

    atom_thresh = parsed_data.thresh*parsed_data.countsperphoton
    binarized = np.clip(roisums, atom_thresh, atom_thresh+1) - atom_thresh        
    # binarized: (npair, nrep, pair, rois)
    fill_first = np.mean(binarized[:,:,0,:], axis = (1,2))
    fill_last = np.mean(binarized[:,:,-1,:], axis = (1,2))
    fill_rois = np.mean(binarized[:,:,0,:], axis = 1)
    fillfracs = np.mean(binarized, axis = (1,2,3))
#     print(tuple(np.arange(1, binarized.ndim)))

    z = 1.96 # corresponds to 95% CI
    error_first = z*np.sqrt(fill_first*(1-fill_first)/parsed_data.points)
    error = z*np.sqrt(fillfracs*(1-fillfracs)/parsed_data.points)
    error_last = z*np.sqrt(fill_last*(1-fill_last)/parsed_data.points)

    pairs = np.arange(parsed_data.npics//2)
    popt, pcov = curve_fit(expfit, pairs, fill_first)

    plt.errorbar(range(parsed_data.npics//2), fill_first, yerr = error_first, fmt='k-', zorder = 1)
    plt.errorbar(range(parsed_data.npics//2), fillfracs, yerr = error, fmt='-', color="0.5", zorder = 0)
    plt.errorbar(range(parsed_data.npics//2), fill_last, yerr = error_last, fmt='-', color="0.75", zorder = 0)
    plt.ylabel('Fill Fraction', fontsize = 14)
    plt.xlabel("Cycle number", fontsize = 14)
    plt.xticks(range(roisums.shape[0]))
    plt.title("File " + str(parsed_data.fnum))
    plt.legend(['1st image', 'Average', 'Last'])
    plt.plot(pairs, expfit(pairs, *popt))
    plt.show()
    print('decay rate: ' + str(popt[1]) + ' +- ' + str(np.sqrt(pcov[1][1])))
    if plotROIs:
        for i in range(len(rois)):
            fill_roi = fill_rois[:,i]
            error_roi = z*np.sqrt(fill_roi*(1-fill_roi)/parsed_data.points)
            plt.errorbar(range(parsed_data.pairsPerRep), fill_roi, yerr = error_roi, fmt='-', zorder = 1)
        plt.ylabel('Fill Fraction', fontsize = 14)
        plt.xlabel("Cycle number", fontsize = 14)
        plt.xticks(range(roisums.shape[0]))
        plt.title("File " + str(parsed_data.fnum))
        plt.legend(range(len(parsed_data.rois)))
        plt.show()
        
def plotIndLosses(parsed_data, plot = True):
    cents = []
    cents_err = []
    amp = []
    amp_err = []

    for roinum in range(len(parsed_data.rois)):
        losses =[]
        losserrs = []
        infids = []
        for var in range(roisums.shape[0]):
            infidelity, inf_err, lossfrac, loss_err = hist_stats(parsed_data.roisums,i=var, thresh = parsed_data.thresh, countsperphoton = parsed_data.countsperphoton, roinum = roinum, quiet = True, plots = False)
            losses.append(lossfrac)
            infids.append(infidelity)
            losserrs.append(loss_err)
#             maxcoord = ks[np.argmax(losses)]
        losses = arr(losses)        
        ks = arr(parsed_data.keySort)
        losserrs = arr(losserrs)        
        ks = ks[losses>0]
        losserrs = losserrs[losses>0]
        losses = losses[losses>0]
#             ks = ks[:-4]
#             losserrs = losserrs[:-4]
#             losses = losses[:-4]

# a0, a1, a2, sigma_carrier, sigma_side, f_carrier, df_sidebands, y0
#             guess = [.05, .1, .05, .1, .1, 80.1, .1, 0]
#             popt, pcov = curve_fit(triplor, ks, -(losses-100)/100, p0 = guess, maxfev=100000, bounds = (0,[1,1,1,1,1,np.inf,1,1]))
#             perr = np.sqrt(np.diag(pcov))

        xs = np.linspace(np.min(ks), np.max(ks), 100)

        try:
            popt, perr = gausfit(ks, (100-losses)/100, y_offset=True, negative=False)
            xs = np.linspace(np.min(ks), np.max(ks), 100)
            cents.append(popt[0])
            cents_err.append(perr[0])
            amp.append(popt[1])
            amp_err.append(perr[1])
#                 plt.plot(xs, gaussian(xs, *popt), 'r-')
        except RuntimeError:
            cents.append(0)
            cents_err.append(0)
            amp.append(0)
            amp_err.append(0)

#             plt.plot(xs, triplor(xs, *popt), 'r-')

#             plt.errorbar(ks, (100-losses)/100, yerr=losserrs/100)
#             plt.xlabel(parsed_data.keyName)
#             plt.ylabel('losses')
#             plt.axis([min(xs), max(xs), 0, 1.1*max(-(losses-100)/100)])
#             plt.title(roinum)

#             if plot:
#                 plt.show()
#             else:
#                 plt.close()
    return cents, cents_err, amp, amp_err

def plotROILosses(parsed_data, plotTransfer = True, ptnum = 0, masks = False):
    losses =[]
    losserrs = []
    infids = []

    if masks:
        n = np.sqrt(len(parsed_data.masks)).astype(int)
    else:
        n = np.sqrt(len(parsed_data.rois)).astype(int)

    for roinum in range(n**2):
        infidelity, inf_err, lossfrac, loss_err = hist_stats(parsed_data.roisums,i=ptnum, thresh = parsed_data.thresh, countsperphoton = parsed_data.countsperphoton, roinum = roinum, quiet = True, plots = False)
        losses.append(lossfrac)
        infids.append(infidelity)
        losserrs.append(loss_err)
#             maxcoord = ks[np.argmax(losses)]
    losses = np.reshape(arr(losses), (n, n))  
    losserrs = np.reshape(arr(losserrs), (n, n)) 

    if not plotTransfer:
        plt.imshow(losses, aspect ='auto')
        plt.colorbar()
        plt.title("File " + str(parsed_data.fnum))
        plt.show()
    else:
        transfer = 100-losses
        plt.imshow(transfer, aspect ='auto', vmin = np.min(transfer), vmax = np.max(transfer))
#             plt.imshow(losses, aspect ='auto', vmin = 0, vmax = 100)

        plt.colorbar()
#             plt.title("File " + str(parsed_data.fnum) + ', transfer')
        plt.title('Variation ' + str(ptnum))
        plt.show()

def plotTimeLosses(parsed_data):
    cycs, losses, losserrs = parsed_data.getLossData(timeOrdered = True)
    plt.errorbar(cycs, losses, yerr=losserrs, fmt = 'r-')
    plt.xlabel('Cycle number')
    plt.ylabel('Losses')
    plt.title("File " + str(parsed_data.fnum))
    plt.axis([cycs[0], cycs[-1], 0, 100])
    plt.xticks(cycs)
    plt.show()

def fitIndividualData(parsed_data):
    centers = []

    for roinum in range(len(rois)):
        losses = []
        losserrs = []
        infids = []

        for var in range(roisums.shape[0]):
                infidelity, inf_err, lossfrac, loss_err = hist_stats(parsed_data.roisums,i=var, thresh = parsed_data.thresh, countsperphoton = parsed_data.countsperphoton, roinum = roinum, quiet = True, plots = False)
                losses.append(lossfrac)
                infids.append(infidelity)
                losserrs.append(loss_err)
        plt.errorbar(parsed_data.keySort, losses, yerr=losserrs)
        plt.xlabel('scan parameter')
        plt.ylabel('losses')
        plt.axis([np.min(parsed_data.keySort), np.max(parsed_data.keySort), 0, 100])
        plt.show()
#             p_guess = [ks[np.argmax(losses)], np.max(losses), .0001, 0]

#             p_guess = [np.max(losses)/2, np.max(losses), np.max(losses)/2, .1, .1, ks[np.argmax(losses)], 0.1, 0]

        guess = [.05, .1, .05, .1, .1, 80.1, .1, 0]
        popt, pcov = curve_fit(triplor, ks, -(losses-100)/100, p0 = guess, maxfev=100000, bounds = (0,[1,1,1,1,1,np.inf,1,1]))
        perr = np.sqrt(np.diag(pcov))

#             popt, pcov = curve_fit(gaussian, ks, losses, p0 = p_guess, maxfev=100000)
#             popt, perr = gausfit(ks, losses,y_offset=True, negative=True)

        xs = np.linspace(np.min(ks), np.max(ks), 100)
        plt.plot(ks, losses, 'k-')
        plt.plot(ks, losses, 'k.')
        plt.plot(xs, gaussian(xs, *popt), 'r-')
        plt.xlabel(data.keyName)
        plt.ylabel('losses')

        plt.axvline(x=popt[0], ymin=0, ymax = 100, linewidth=2, color='b')

        print("Current file: ", current)
        plt.show()

        print('center frequency: ' + str(popt[0]) + ' MHz')
        print('width: ' + str(1000000*popt[2]) + 'Hz')
        centers.append(popt[0]*1000000)
    print(centers)
    plt.plot(range(len(centers)), centers-np.mean(centers), 'ko')
    plt.xlabel('trap number')
    plt.ylabel('center deviation (Hz)')

def fitSpot(parsed_data, magnification = ((16*10**-6)*(22*10**-3)/(1000*10**-3)*10**9)):
    img = parsed_data.roiimg
    fitdat, params, perr = gaussianBeamFit2D(img)
    plt.imshow(img)
    plt.contour(fitdat)
    plt.show()
    print("Waists (nm): " + str(params[-3:-1]*magnification))

def plotIndSpect(parsed_data, plot = True):
    cents = []
    cents_err = []
    df = []
    df_err = []
    nbar = []
    nbar_err = []
    for roinum in range(len(parsed_data.rois)):
        losses =[]
        losserrs = []
        infids = []
        for var in range(roisums.shape[0]):
            infidelity, inf_err, lossfrac, loss_err = hist_stats(parsed_data.roisums,i=var, thresh = parsed_data.thresh, countsperphoton = parsed_data.countsperphoton, roinum = roinum, quiet = True, plots = False)
            losses.append(lossfrac)
            infids.append(infidelity)
            losserrs.append(loss_err)
#             maxcoord = ks[np.argmax(losses)]
        losses = arr(losses)
        ks = arr(parsed_data.keySort)
        losserrs = arr(losserrs)
        ks = ks[losses>0]
        losserrs = losserrs[losses>0]
        losses = losses[losses>0]

# a0, a1, a2, sigma_carrier, sigma_side, f_carrier, df_sidebands, y0
#             guess = [.05, .1, .05, .1, .1, 80.1, .1, 0]
        guess = [.2, .2, .1, .1, .1, 80.015, 0.1, 0]
        popt, pcov = curve_fit(triplor, ks, -(losses-100)/100, p0 = guess, maxfev=100000, bounds = (0,[1,1,1,1,1,np.inf,1,1]))
        perr = np.sqrt(np.diag(pcov))

#             popt, perr = gausfit(ks, losses,y_offset=True, negative=True)
        xs = np.linspace(np.min(ks), np.max(ks), 100)
#             plt.plot(xs, gaussian(xs, *popt), 'r-')
#             centers.append(popt[0])
#             errors.append(perr[0])
        cents.append(popt[-3])
        cents_err.append(perr[-3])
        df.append(popt[-2])
        df_err.append(perr[-2])

        nbar.append(1/((popt[1]/popt[0])-1))
        err = (popt[0]/popt[1])*np.sqrt(perr[0]**2+perr[1]**2)
        nbar_err.append(1/err)

        plt.plot(xs, triplor(xs, *popt), 'r-')
        plt.errorbar(ks, -(losses-100)/100, yerr=losserrs/100)
        plt.xlabel(parsed_data.keyName)
        plt.ylabel('losses')
        plt.axis([min(xs), max(xs), 0, 1.1*max(-(losses-100)/100)])
        plt.title(roinum)

        if plot:
            plt.show()
        else:
            plt.close()
    return cents, cents_err, df, df_err, nbar, nbar_err