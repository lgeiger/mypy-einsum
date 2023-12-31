from array_module import einsum as einsum
import numpy as np

A: np.ndarray

einsum("", 0)
einsum("...", A)
einsum("ij", A)
einsum("ji", A)
einsum("ii->i", A)
einsum("...ii->...i", A)
einsum("ii...->...i", A)
einsum("...ii->i...", A)
einsum("jii->ij", A)
einsum("ii...->i...", A)
einsum("i...i->i...", A)
einsum("i...i->...i", A)
einsum("iii->i", A)
einsum("ijk->jik", A)

einsum("i->", A)
einsum("...i->...", A)
einsum("i...->...", A)
einsum("ii", A)

einsum("..., ...", A, A)
einsum("...i, ...i", A, A)
einsum("i..., i...", A, A)
einsum("i, i", A, A)
einsum("ij, j", A, A)
einsum("ji, j", A, A)
einsum("ij,jk", A, A)
einsum("ij,jk,kl", A, A, A)
einsum("ijk, jil -> kl", A, A)
einsum("ijk,jil->kl", A, A)
einsum("i,i,i->i", A, A, A)
einsum("i,->", A, 3)
einsum("...,...", A, A)
einsum("i,i", A, A)
einsum("i,->i", A, 2)
einsum(",i->i", 2, A)
einsum(",i->", 2, A)
einsum("z,mz,zm->", A, A, A)
einsum("ij,ij->j", A, A)
einsum("...ij,...jk->...ik", A, A)
einsum("ji,i->", A, A)
einsum("i,ij->", A, A)
einsum("ij,i->", A, A)

einsum("ij...,j...->i...", A, A)
einsum("...i,...i", A, A)
einsum("ijklm,ijn,ijn->", A, A, A)
einsum("ijklm,ijn->", A, A)
einsum("x,yx,zx->xzy", A, A, A)

einsum("ijk,j->ijk", A, A)
einsum("ij...,j...->ij...", A, A)
einsum("ij...,...j->ij...", A, A)
einsum("ij...,j->ij...", A, A)
einsum("ik,kj->ij", A, A)
einsum("ik...,k...->i...", A, A)
einsum("ik...,...kj->i...j", A, A)
einsum("...k,kj", A, A)
einsum("ik,k...->i...", A, A)
einsum("ijkl,k->ijl", A, A)
einsum("ijkl,k", A, A)
einsum("...kl,k", A, A)
einsum("...kl,k...", A, A)
einsum("...lmn,...lmno->...o", A, A)
einsum("...lmn,lmno->...o", A, A)

einsum("cl, cpx->lpx", A, A)
einsum("cl, cpxy->lpxy", A, A)
einsum("aabb->ab", A)

einsum("ijij->", A)

einsum("mi,mi,mi->m", A, A, A)
einsum("im,im,im->m", A, A, A)
einsum("ij,jk->ik", A, A)

einsum("i,i->i", A, A)
einsum("i,i->", A, A)
einsum("i,i,i->", A, A, A)
einsum("i,i,i,i->", A, A, A, A)

einsum("...ij,...jk->...ik", A, A)
