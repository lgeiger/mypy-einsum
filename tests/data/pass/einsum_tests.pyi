import numpy as np

A: np.ndarray

np.einsum("", 0)
np.einsum("...", A)
np.einsum("ij", A)
np.einsum("ji", A)
np.einsum("ii->i", A)
np.einsum("...ii->...i", A)
np.einsum("ii...->...i", A)
np.einsum("...ii->i...", A)
np.einsum("jii->ij", A)
np.einsum("ii...->i...", A)
np.einsum("i...i->i...", A)
np.einsum("i...i->...i", A)
np.einsum("iii->i", A)
np.einsum("ijk->jik", A)

np.einsum("i->", A)
np.einsum("...i->...", A)
np.einsum("i...->...", A)
np.einsum("ii", A)

np.einsum("..., ...", A, A)
np.einsum("...i, ...i", A, A)
np.einsum("i..., i...", A, A)
np.einsum("i, i", A, A)
np.einsum("ij, j", A, A)
np.einsum("ji, j", A, A)
np.einsum("ij,jk", A, A)
np.einsum("ij,jk,kl", A, A, A)
np.einsum("ijk, jil -> kl", A, A)
np.einsum("ijk,jil->kl", A, A)
np.einsum("i,i,i->i", A, A, A)
np.einsum("i,->", A, 3)
np.einsum("...,...", A, A)
np.einsum("i,i", A, A)
np.einsum("i,->i", A, 2)
np.einsum(",i->i", 2, A)
np.einsum(",i->", 2, A)
np.einsum("z,mz,zm->", A, A, A)
np.einsum("ij,ij->j", A, A)
np.einsum("...ij,...jk->...ik", A, A)
np.einsum("ji,i->", A, A)
np.einsum("i,ij->", A, A)
np.einsum("ij,i->", A, A)

np.einsum("ij...,j...->i...", A, A)
np.einsum("...i,...i", A, A)
np.einsum("ijklm,ijn,ijn->", A, A, A)
np.einsum("ijklm,ijn->", A, A)
np.einsum("x,yx,zx->xzy", A, A, A)

np.einsum("ijk,j->ijk", A, A)
np.einsum("ij...,j...->ij...", A, A)
np.einsum("ij...,...j->ij...", A, A)
np.einsum("ij...,j->ij...", A, A)
np.einsum("ik,kj->ij", A, A)
np.einsum("ik...,k...->i...", A, A)
np.einsum("ik...,...kj->i...j", A, A)
np.einsum("...k,kj", A, A)
np.einsum("ik,k...->i...", A, A)
np.einsum("ijkl,k->ijl", A, A)
np.einsum("ijkl,k", A, A)
np.einsum("...kl,k", A, A)
np.einsum("...kl,k...", A, A)
np.einsum("...lmn,...lmno->...o", A, A)
np.einsum("...lmn,lmno->...o", A, A)

np.einsum("cl, cpx->lpx", A, A)
np.einsum("cl, cpxy->lpxy", A, A)
np.einsum("aabb->ab", A)

np.einsum("ijij->", A)

np.einsum("mi,mi,mi->m", A, A, A)
np.einsum("im,im,im->m", A, A, A)
np.einsum("ij,jk->ik", A, A)

np.einsum("i,i->i", A, A)
np.einsum("i,i->", A, A)
np.einsum("i,i,i->", A, A, A)
np.einsum("i,i,i,i->", A, A, A, A)

np.einsum("...ij,...jk->...ik", A, A)