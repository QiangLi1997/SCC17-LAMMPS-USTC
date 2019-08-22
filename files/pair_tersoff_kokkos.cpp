/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Ray Shan (SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_tersoff_kokkos.h"
#include "kokkos.h"
#include "atom_kokkos.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list_kokkos.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "atom_masks.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define KOKKOS_CUDA_MAX_THREADS 256
#define KOKKOS_CUDA_MIN_BLOCKS 8

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairTersoffKokkos<DeviceType>::PairTersoffKokkos(LAMMPS *lmp) : PairTersoff(lmp)
{
  THIRD = 1.0/3.0;

  respa_enable = 0;

  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairTersoffKokkos<DeviceType>::~PairTersoffKokkos()
{
  if (!copymode) {
    memory->destroy_kokkos(k_eatom,eatom);
    memory->destroy_kokkos(k_vatom,vatom);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairTersoffKokkos<DeviceType>::allocate()
{
  PairTersoff::allocate();

  int n = atom->ntypes;

  k_params = Kokkos::DualView<params_ters***,Kokkos::LayoutRight,DeviceType>
	  ("PairTersoff::paramskk",n+1,n+1,n+1);
  paramskk = k_params.d_view;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairTersoffKokkos<DeviceType>::init_style()
{
  PairTersoff::init_style();

  // irequest = neigh request made by parent class

  neighflag = lmp->kokkos->neighflag;
  int irequest = neighbor->nrequest - 1;

  neighbor->requests[irequest]->
    kokkos_host = Kokkos::Impl::is_same<DeviceType,LMPHostType>::value &&
    !Kokkos::Impl::is_same<DeviceType,LMPDeviceType>::value;
  neighbor->requests[irequest]->
    kokkos_device = Kokkos::Impl::is_same<DeviceType,LMPDeviceType>::value;

  if (neighflag == FULL || neighflag == HALF || neighflag == HALFTHREAD) {
  //if (neighflag == FULL || neighflag == HALFTHREAD) {
    neighbor->requests[irequest]->full = 1;
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full_cluster = 0;
    if (neighflag == FULL)
      neighbor->requests[irequest]->ghost = 1;
    else
      neighbor->requests[irequest]->ghost = 0;
  } else {
    error->all(FLERR,"Cannot use chosen neighbor list style with tersoff/kk");
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairTersoffKokkos<DeviceType>::setup_params()
{
  PairTersoff::setup_params();

  int i,j,k,m;
  int n = atom->ntypes;

  for (i = 1; i <= n; i++)
    for (j = 1; j <= n; j++)
      for (k = 1; k <= n; k++) {
	m = elem2param[i-1][j-1][k-1];
	k_params.h_view(i,j,k).powerm = params[m].powerm;
	k_params.h_view(i,j,k).gamma = params[m].gamma;
	k_params.h_view(i,j,k).lam3 = params[m].lam3;
	k_params.h_view(i,j,k).c = params[m].c;
	k_params.h_view(i,j,k).d = params[m].d;
	k_params.h_view(i,j,k).h = params[m].h;
	k_params.h_view(i,j,k).powern = params[m].powern;
	k_params.h_view(i,j,k).beta = params[m].beta;
	k_params.h_view(i,j,k).lam2 = params[m].lam2;
	k_params.h_view(i,j,k).bigb = params[m].bigb;
	k_params.h_view(i,j,k).bigr = params[m].bigr;
	k_params.h_view(i,j,k).bigd = params[m].bigd;
	k_params.h_view(i,j,k).lam1 = params[m].lam1;
	k_params.h_view(i,j,k).biga = params[m].biga;
	k_params.h_view(i,j,k).cutsq = params[m].cutsq;
	k_params.h_view(i,j,k).c1 = params[m].c1;
	k_params.h_view(i,j,k).c2 = params[m].c2;
	k_params.h_view(i,j,k).c3 = params[m].c3;
	k_params.h_view(i,j,k).c4 = params[m].c4;
      }

  k_params.template modify<LMPHostType>();

}

/* ---------------------------------------------------------------------- */
#include<fenv.h>

template<class DeviceType>
void PairTersoffKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;
feenableexcept(FE_INVALID|FE_OVERFLOW);

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memory->destroy_kokkos(k_eatom,eatom);
    memory->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.d_view;
  }
  if (vflag_atom) {
    memory->destroy_kokkos(k_vatom,vatom);
    memory->create_kokkos(k_vatom,vatom,maxvatom,6,"pair:vatom");
    d_vatom = k_vatom.d_view;
  }

  atomKK->sync(execution_space,datamask_read);
  k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;

  const int inum = list->inum;
  const int ignum = inum + list->gnum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  k_list->clean_copy();
  copymode = 1;

  EV_FLOAT ev;
  EV_FLOAT ev_all;

  ntypes = atom->ntypes;

  if (neighflag == HALF) {
    //if (evflag)
    //  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairTersoffComputeHalf<HALF,1> >(0,inum),*this,ev);
    //else
    //  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairTersoffComputeHalf<HALF,0> >(0,inum),*this);
    //DeviceType::fence();
    //ev_all += ev;
  } else if (neighflag == HALFTHREAD) {
    if (evflag)
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairTersoffComputeHalf<HALFTHREAD,1> >(0,inum),*this,ev);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairTersoffComputeHalf<HALFTHREAD,0> >(0,inum),*this);
    DeviceType::fence();
    ev_all += ev;
  } else if (neighflag == FULL) {
    //assert(false);
    //if (evflag)
    //  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairTersoffComputeFullA<FULL,1> >(0,inum),*this,ev);
    //else
    //  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairTersoffComputeFullA<FULL,0> >(0,inum),*this);
    //DeviceType::fence();
    //ev_all += ev;

    //if (evflag)
    //  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairTersoffComputeFullB<FULL,1> >(0,ignum),*this,ev);
    //else
    //  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairTersoffComputeFullB<FULL,0> >(0,ignum),*this);
    //DeviceType::fence();
    //ev_all += ev;
  }

  if (eflag_global) eng_vdwl += ev_all.evdwl;
  if (vflag_global) {
    virial[0] += ev_all.v[0];
    virial[1] += ev_all.v[1];
    virial[2] += ev_all.v[2];
    virial[3] += ev_all.v[3];
    virial[4] += ev_all.v[4];
    virial[5] += ev_all.v[5];
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  copymode = 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairTersoffKokkos<DeviceType>::operator()(TagPairTersoffComputeHalf<NEIGHFLAG,EVFLAG>, const int &ii, EV_FLOAT& ev) const {

  // The f array is atomic for Half/Thread neighbor style
  Kokkos::View<F_FLOAT*[3], typename DAT::t_f_array::array_layout,DeviceType,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > a_f = f;

  const int i = d_ilist[ii];
  if (i >= nlocal) return;
  const X_FLOAT xtmp = x(i,0);
  const X_FLOAT ytmp = x(i,1);
  const X_FLOAT ztmp = x(i,2);
  const int itype = type(i);
  const int itag = tag(i);

  int j,k,jj,kk,jtag,jtype,ktype;
  F_FLOAT rsq1, cutsq1, rsq2, cutsq2, rij, rik, bo_ij;
  F_FLOAT fi[3], fj[3], fk[3];
  X_FLOAT delx1, dely1, delz1, delx2, dely2, delz2;

  //const AtomNeighborsConst d_neighbors_i = k_list.get_neighbors_const(i);
  const int jnum = d_numneigh[i];

  int js[16];
  bool js_repulsive_mask[16];
  int jsnum = 0;
  bool repulsive;

  for (jj = 0; jj < jnum; jj++) {
    j = d_neighbors(i,jj);
    j &= NEIGHMASK;
    jtype = type(j);
    jtag = tag(j);

    const X_FLOAT delx = xtmp - x(j,0);
    const X_FLOAT dely = ytmp - x(j,1);
    const X_FLOAT delz = ztmp - x(j,2);
    const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;
    const F_FLOAT cutsq = paramskk(itype,jtype,jtype).cutsq;

    if (rsq > cutsq) continue;
    js[jsnum] = j;
    js_repulsive_mask[jsnum] = false;
    jsnum += 1;
    if (itag > jtag) {
      if ((itag+jtag) % 2 == 0) continue;
    } else if (itag < jtag) {
      if ((itag+jtag) % 2 == 1) continue;
    } else {
      if (x(j,2)  < ztmp) continue;
      if (x(j,2) == ztmp && x(j,1)  < ytmp) continue;
      if (x(j,2) == ztmp && x(j,1) == ytmp && x(j,0) < xtmp) continue;
    }
    js_repulsive_mask[jsnum-1] = true;
  }
  int jj_after = jj;
  int jj_remainder = jnum - jj + jsnum;

  // attractive: bond order

  for (jj = 0; jj < jj_remainder; jj++) {
    if (jj < jsnum) {
      j = js[jj];
      repulsive = js_repulsive_mask[jj];
    } else {
      j = d_neighbors(i,jj - jsnum + jj_after);
      j &= NEIGHMASK;
      jtype = type(j);
      jtag = tag(j);

      const X_FLOAT delx = xtmp - x(j,0);
      const X_FLOAT dely = ytmp - x(j,1);
      const X_FLOAT delz = ztmp - x(j,2);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;
      const F_FLOAT cutsq = paramskk(itype,jtype,jtype).cutsq;

      if (rsq > cutsq) continue;
      repulsive = false;
      if (itag > jtag) {
        if ((itag+jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
        if ((itag+jtag) % 2 == 1) continue;
      } else {
        if (x(j,2)  < ztmp) continue;
        if (x(j,2) == ztmp && x(j,1)  < ytmp) continue;
        if (x(j,2) == ztmp && x(j,1) == ytmp && x(j,0) < xtmp) continue;
      }
      repulsive = true;
    }
    int is[1], js[1];
    is[0] = i; js[0] = j;
    kernel_step<EVFLAG,NEIGHFLAG>(1, is, js, repulsive, ev);
//    continue;
//    jtype = type(j);
//
//    delx1 = xtmp - x(j,0);
//    dely1 = ytmp - x(j,1);
//    delz1 = ztmp - x(j,2);
//    rsq1 = delx1*delx1 + dely1*dely1 + delz1*delz1;
//    cutsq1 = paramskk(itype,jtype,jtype).cutsq;
//
//    bo_ij = 0.0;
//    if (rsq1 > cutsq1) continue;
//    rij = sqrt(rsq1);
//
//#ifndef __CUDA_ARCH__
//#define __any(a) (a)
//#define __all(a) (a)
//#endif
//    bool active, working;
//    active = true;
//    kk = 0;
//    active = kk < jnum;
//    while (__any(active)) {
//      if (active) k = d_neighbors(i,kk);
//      k &= NEIGHMASK;
//      working = active;
//      if (j == k) working = false;
//      ktype = type(k);
//      delx2 = xtmp - x(k,0);
//      dely2 = ytmp - x(k,1);
//      delz2 = ztmp - x(k,2);
//      rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
//      cutsq2 = paramskk(itype,jtype,ktype).cutsq;
//      if (rsq2 > cutsq2) working = false;
//      if (__all(working || ! active)) {
//        if (working && active) {
//          rik = sqrt(rsq2);
//          bo_ij += bondorder(itype,jtype,ktype,rij,delx1,dely1,delz1,rik,delx2,dely2,delz2);
//        }
//        kk += 1;
//      } else {
//        if (! working && active) { kk += 1; }
//      }
//      active = kk < jnum;
//    }
//
//    for (kk = 0; 0&&kk < jnum; kk++) {
//      k = d_neighbors(i,kk);
//      k &= NEIGHMASK;
//      if (j == k) continue;
//      ktype = type(k);
//
//      delx2 = xtmp - x(k,0);
//      dely2 = ytmp - x(k,1);
//      delz2 = ztmp - x(k,2);
//      rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
//      cutsq2 = paramskk(itype,jtype,ktype).cutsq;
//
//      if (rsq2 > cutsq2) continue;
//      rik = sqrt(rsq2);
//      bo_ij += bondorder(itype,jtype,ktype,rij,delx1,dely1,delz1,rik,delx2,dely2,delz2);
//    }
//    // attractive: pairwise potential and force
//
//    const F_FLOAT fa = ters_fa_k(itype,jtype,jtype,rij);
//    const F_FLOAT dfa = ters_dfa(itype,jtype,jtype,rij);
//    const F_FLOAT bij = ters_bij_k(itype,jtype,jtype,bo_ij);
//    const F_FLOAT fatt = -0.5*bij * dfa / rij;
//    const F_FLOAT prefactor = 0.5*fa * ters_dbij(itype,jtype,jtype,bo_ij);
//    const F_FLOAT eng_fa = 0.5*bij * fa;
//    
//    const F_FLOAT tmp_fce = ters_fc_k(itype,jtype,jtype,rij);
//    const F_FLOAT tmp_fcd = ters_dfc(itype,jtype,jtype,rij);
//    const F_FLOAT tmp_exp = exp(-paramskk(itype,jtype,jtype).lam1 * rij);
//    F_FLOAT frep = -paramskk(itype,jtype,jtype).biga * tmp_exp *
//	    		  (tmp_fcd - tmp_fce*paramskk(itype,jtype,jtype).lam1) / rij;
//    F_FLOAT eng_fr = tmp_fce * paramskk(itype,jtype,jtype).biga * tmp_exp;
//
//    if (! repulsive) { frep = 0; eng_fr = 0; }
//
//    const F_FLOAT fpair = frep + fatt;
//    const F_FLOAT eng = eng_fa + eng_fr;
//
//    a_f(i,0) += delx1*fpair;
//    a_f(i,1) += dely1*fpair;
//    a_f(i,2) += delz1*fpair;
//    a_f(j,0) -= delx1*fpair;
//    a_f(j,1) -= dely1*fpair;
//    a_f(j,2) -= delz1*fpair;
//
//    if (EVFLAG) {
//      if (eflag) ev.evdwl += eng;
//      if (vflag_either || eflag_atom)
//	this->template ev_tally<NEIGHFLAG>(ev,i,j,eng,fpair,delx1,dely1,delz1);
//    }
//
//    // attractive: three-body force
//
//    active = true;
//    kk = 0;
//    active = kk < jnum;
//    while (__any(active)) {
//      if (active) k = d_neighbors(i,kk);
//      k &= NEIGHMASK;
//      working = active;
//      if (j == k) working = false;
//      ktype = type(k);
//      delx2 = xtmp - x(k,0);
//      dely2 = ytmp - x(k,1);
//      delz2 = ztmp - x(k,2);
//      rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
//      cutsq2 = paramskk(itype,jtype,ktype).cutsq;
//      if (rsq2 > cutsq2) working = false;
//      if (__all(working || ! active)) {
//        if (working && active) {
//          rik = sqrt(rsq2);
//          ters_dthb(itype,jtype,ktype,prefactor,rij,delx1,dely1,delz1,
//            	rik,delx2,dely2,delz2,fi,fj,fk);
//
//          a_f(i,0) += fi[0];
//          a_f(i,1) += fi[1];
//          a_f(i,2) += fi[2];
//          a_f(j,0) += fj[0];
//          a_f(j,1) += fj[1];
//          a_f(j,2) += fj[2];
//          a_f(k,0) += fk[0];
//          a_f(k,1) += fk[1];
//          a_f(k,2) += fk[2];
//
//          if (vflag_atom) {
//            F_FLOAT delrij[3], delrik[3];
//            delrij[0] = -delx1; delrij[1] = -dely1; delrij[2] = -delz1;
//            delrik[0] = -delx2; delrik[1] = -dely2; delrik[2] = -delz2;
//            if (vflag_either) this->template v_tally3<NEIGHFLAG>(ev,i,j,k,fj,fk,delrij,delrik);
//          }
//        }
//        kk += 1;
//      } else {
//        if (! working && active) { kk += 1; }
//      }
//      active = kk < jnum;
//    }
//    for (kk = 0;0&& kk < jnum; kk++) {
//      k = d_neighbors(i,kk);
//      k &= NEIGHMASK;
//      if (j == k) continue;
//      ktype = type(k);
//
//      delx2 = xtmp - x(k,0);
//      dely2 = ytmp - x(k,1);
//      delz2 = ztmp - x(k,2);
//      rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
//      cutsq2 = paramskk(itype,jtype,ktype).cutsq;
//
//      if (rsq2 > cutsq2) continue;
//      rik = sqrt(rsq2);
//      ters_dthb(itype,jtype,ktype,prefactor,rij,delx1,dely1,delz1,
//        	rik,delx2,dely2,delz2,fi,fj,fk);
//
//      a_f(i,0) += fi[0];
//      a_f(i,1) += fi[1];
//      a_f(i,2) += fi[2];
//      a_f(j,0) += fj[0];
//      a_f(j,1) += fj[1];
//      a_f(j,2) += fj[2];
//      a_f(k,0) += fk[0];
//      a_f(k,1) += fk[1];
//      a_f(k,2) += fk[2];
//
//      if (vflag_atom) {
//        F_FLOAT delrij[3], delrik[3];
//        delrij[0] = -delx1; delrij[1] = -dely1; delrij[2] = -delz1;
//        delrik[0] = -delx2; delrik[1] = -dely2; delrik[2] = -delz2;
//        if (vflag_either) this->template v_tally3<NEIGHFLAG>(ev,i,j,k,fj,fk,delrij,delrik);
//      }
//    }
  }
}

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairTersoffKokkos<DeviceType>::operator()(TagPairTersoffComputeHalf<NEIGHFLAG,EVFLAG>, const int &ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairTersoffComputeHalf<NEIGHFLAG,EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairTersoffKokkos<DeviceType>::ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                const F_FLOAT &dely, const F_FLOAT &delz) const
{
  const int VFLAG = vflag_either;

  // The eatom and vatom arrays are atomic for Half/Thread neighbor style
  Kokkos::View<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,DeviceType,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_eatom = k_eatom.view<DeviceType>();
  Kokkos::View<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,DeviceType,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_vatom = k_vatom.view<DeviceType>();

  if (eflag_atom) {
    const E_FLOAT epairhalf = 0.5 * epair;
    v_eatom[i] += epairhalf;
    if (NEIGHFLAG != FULL) v_eatom[j] += epairhalf;
  }

  if (VFLAG) {
    const E_FLOAT v0 = delx*delx*fpair;
    const E_FLOAT v1 = dely*dely*fpair;
    const E_FLOAT v2 = delz*delz*fpair;
    const E_FLOAT v3 = delx*dely*fpair;
    const E_FLOAT v4 = delx*delz*fpair;
    const E_FLOAT v5 = dely*delz*fpair;

    if (vflag_global) {
      if (NEIGHFLAG != FULL) {
        ev.v[0] += v0;
        ev.v[1] += v1;
        ev.v[2] += v2;
        ev.v[3] += v3;
        ev.v[4] += v4;
        ev.v[5] += v5;
      } else {
        ev.v[0] += 0.5*v0;
        ev.v[1] += 0.5*v1;
        ev.v[2] += 0.5*v2;
        ev.v[3] += 0.5*v3;
        ev.v[4] += 0.5*v4;
        ev.v[5] += 0.5*v5;
      }
    }

    if (vflag_atom) {
      v_vatom(i,0) += 0.5*v0;
      v_vatom(i,1) += 0.5*v1;
      v_vatom(i,2) += 0.5*v2;
      v_vatom(i,3) += 0.5*v3;
      v_vatom(i,4) += 0.5*v4;
      v_vatom(i,5) += 0.5*v5;

      if (NEIGHFLAG != FULL) {
        v_vatom(j,0) += 0.5*v0;
        v_vatom(j,1) += 0.5*v1;
        v_vatom(j,2) += 0.5*v2;
        v_vatom(j,3) += 0.5*v3;
        v_vatom(j,4) += 0.5*v4;
        v_vatom(j,5) += 0.5*v5;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairTersoffKokkos<DeviceType>::v_tally3(EV_FLOAT &ev, const int &i, const int &j, const int &k,
	F_FLOAT *fj, F_FLOAT *fk, F_FLOAT *drij, F_FLOAT *drik) const
{

  // The eatom and vatom arrays are atomic for Half/Thread neighbor style
  Kokkos::View<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,DeviceType,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_vatom = k_vatom.view<DeviceType>();

  F_FLOAT v[6];

  v[0] = THIRD * (drij[0]*fj[0] + drik[0]*fk[0]);
  v[1] = THIRD * (drij[1]*fj[1] + drik[1]*fk[1]);
  v[2] = THIRD * (drij[2]*fj[2] + drik[2]*fk[2]);
  v[3] = THIRD * (drij[0]*fj[1] + drik[0]*fk[1]);
  v[4] = THIRD * (drij[0]*fj[2] + drik[0]*fk[2]);
  v[5] = THIRD * (drij[1]*fj[2] + drik[1]*fk[2]);

  if (vflag_global) {
    ev.v[0] += v[0];
    ev.v[1] += v[1];
    ev.v[2] += v[2];
    ev.v[3] += v[3];
    ev.v[4] += v[4];
    ev.v[5] += v[5];
  }

  if (vflag_atom) {
    v_vatom(i,0) += v[0]; v_vatom(i,1) += v[1]; v_vatom(i,2) += v[2];
    v_vatom(i,3) += v[3]; v_vatom(i,4) += v[4]; v_vatom(i,5) += v[5];
    if (NEIGHFLAG != FULL) {
      v_vatom(j,0) += v[0]; v_vatom(j,1) += v[1]; v_vatom(j,2) += v[2];
      v_vatom(j,3) += v[3]; v_vatom(j,4) += v[4]; v_vatom(j,5) += v[5];
      v_vatom(k,0) += v[0]; v_vatom(k,1) += v[1]; v_vatom(k,2) += v[2];
      v_vatom(k,3) += v[3]; v_vatom(k,4) += v[4]; v_vatom(k,5) += v[5];
    }
  }

}
/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double PairTersoffKokkos<DeviceType>::ters_fc_k(const int &i, const int &j,
		const int &k, const F_FLOAT &r) const
{
  const F_FLOAT ters_R = paramskk(i,j,k).bigr;
  const F_FLOAT ters_D = paramskk(i,j,k).bigd;

  if (r < ters_R-ters_D) return 1.0;
  if (r > ters_R+ters_D) return 0.0;
  return 0.5*(1.0 - sin(MY_PI2*(r - ters_R)/ters_D));
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double PairTersoffKokkos<DeviceType>::ters_dfc(const int &i, const int &j,
		const int &k, const F_FLOAT &r) const
{
  const F_FLOAT ters_R = paramskk(i,j,k).bigr;
  const F_FLOAT ters_D = paramskk(i,j,k).bigd;

  if (r < ters_R-ters_D) return 0.0;
  if (r > ters_R+ters_D) return 0.0;
  return -(MY_PI4/ters_D) * cos(MY_PI2*(r - ters_R)/ters_D);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double PairTersoffKokkos<DeviceType>::bondorder(const int &i, const int &j, const int &k,
	const F_FLOAT &rij, const F_FLOAT &dx1, const F_FLOAT &dy1, const F_FLOAT &dz1,
	const F_FLOAT &rik, const F_FLOAT &dx2, const F_FLOAT &dy2, const F_FLOAT &dz2) const
{
  F_FLOAT arg, ex_delr;

  const F_FLOAT costheta = (dx1*dx2 + dy1*dy2 + dz1*dz2)/(rij*rik);

  if (int(paramskk(i,j,k).powerm) == 3) arg = pow(paramskk(i,j,k).lam3 * (rij-rik),3.0);
  else arg = paramskk(i,j,k).lam3 * (rij-rik);

  if (arg > 69.0776) ex_delr = 1.e30;
  else if (arg < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(arg);

  return ters_fc_k(i,j,k,rik) * ters_gijk(i,j,k,costheta) * ex_delr;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double PairTersoffKokkos<DeviceType>::
	ters_gijk(const int &i, const int &j, const int &k, const F_FLOAT &cos) const
{
  const F_FLOAT ters_c = paramskk(i,j,k).c * paramskk(i,j,k).c;
  const F_FLOAT ters_d = paramskk(i,j,k).d * paramskk(i,j,k).d;
  const F_FLOAT hcth = paramskk(i,j,k).h - cos;

  return paramskk(i,j,k).gamma*(1.0 + ters_c/ters_d - ters_c/(ters_d+hcth*hcth));
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double PairTersoffKokkos<DeviceType>::
	ters_dgijk(const int &i, const int &j, const int &k, const F_FLOAT &cos) const
{

  const F_FLOAT ters_c = paramskk(i,j,k).c * paramskk(i,j,k).c;
  const F_FLOAT ters_d = paramskk(i,j,k).d * paramskk(i,j,k).d;
  const F_FLOAT hcth = paramskk(i,j,k).h - cos;
  const F_FLOAT numerator = -2.0 * ters_c * hcth;
  const F_FLOAT denominator = 1.0/(ters_d + hcth*hcth);
  return paramskk(i,j,k).gamma * numerator * denominator * denominator;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double PairTersoffKokkos<DeviceType>::ters_fa_k(const int &i, const int &j,
		const int &k, const F_FLOAT &r) const
{
  if (r > paramskk(i,j,k).bigr + paramskk(i,j,k).bigd) return 0.0;
  return -paramskk(i,j,k).bigb * exp(-paramskk(i,j,k).lam2 * r)
	  * ters_fc_k(i,j,k,r);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double PairTersoffKokkos<DeviceType>::ters_dfa(const int &i, const int &j,
		const int &k, const F_FLOAT &r) const
{
  if (r > paramskk(i,j,k).bigr + paramskk(i,j,k).bigd) return 0.0;
  return paramskk(i,j,k).bigb * exp(-paramskk(i,j,k).lam2 * r) *
    (paramskk(i,j,k).lam2 * ters_fc_k(i,j,k,r) - ters_dfc(i,j,k,r));
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double PairTersoffKokkos<DeviceType>::ters_bij_k(const int &i, const int &j,
		const int &k, const F_FLOAT &bo) const
{
  const F_FLOAT tmp = paramskk(i,j,k).beta * bo;
  if (tmp > paramskk(i,j,k).c1) return 1.0/sqrt(tmp);
  if (tmp > paramskk(i,j,k).c2)
    return (1.0 - pow(tmp,-paramskk(i,j,k).powern) / (2.0*paramskk(i,j,k).powern))/sqrt(tmp);
  if (tmp < paramskk(i,j,k).c4) return 1.0;
  if (tmp < paramskk(i,j,k).c3)
    return 1.0 - pow(tmp,paramskk(i,j,k).powern)/(2.0*paramskk(i,j,k).powern);
  return pow(1.0 + pow(tmp,paramskk(i,j,k).powern), -1.0/(2.0*paramskk(i,j,k).powern));
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double PairTersoffKokkos<DeviceType>::ters_dbij(const int &i, const int &j,
		const int &k, const F_FLOAT &bo) const
{
  const F_FLOAT tmp = paramskk(i,j,k).beta * bo;
  if (tmp > paramskk(i,j,k).c1) return paramskk(i,j,k).beta * -0.5*pow(tmp,-1.5);
  if (tmp > paramskk(i,j,k).c2)
    return paramskk(i,j,k).beta * (-0.5*pow(tmp,-1.5) *
           (1.0 - 0.5*(1.0 +  1.0/(2.0*paramskk(i,j,k).powern)) *
           pow(tmp,-paramskk(i,j,k).powern)));
  if (tmp < paramskk(i,j,k).c4) return 0.0;
  if (tmp < paramskk(i,j,k).c3)
    return -0.5*paramskk(i,j,k).beta * pow(tmp,paramskk(i,j,k).powern-1.0);

  const F_FLOAT tmp_n = pow(tmp,paramskk(i,j,k).powern);
  return -0.5 * pow(1.0+tmp_n, -1.0-(1.0/(2.0*paramskk(i,j,k).powern)))*tmp_n / bo;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairTersoffKokkos<DeviceType>::ters_dthb(
	const int &i, const int &j, const int &k, const F_FLOAT &prefactor,
	const F_FLOAT &rij, const F_FLOAT &dx1, const F_FLOAT &dy1, const F_FLOAT &dz1,
	const F_FLOAT &rik, const F_FLOAT &dx2, const F_FLOAT &dy2, const F_FLOAT &dz2,
	F_FLOAT *fi, F_FLOAT *fj, F_FLOAT *fk) const
{
  // from PairTersoff::attractive
  F_FLOAT rij_hat[3],rik_hat[3];
  F_FLOAT rijinv,rikinv;
  F_FLOAT delrij[3], delrik[3];

  delrij[0] = dx1; delrij[1] = dy1; delrij[2] = dz1;
  delrik[0] = dx2; delrik[1] = dy2; delrik[2] = dz2;

  //rij = sqrt(rsq1);
  rijinv = 1.0/rij;
  vec3_scale(rijinv,delrij,rij_hat);

  //rik = sqrt(rsq2);
  rikinv = 1.0/rik;
  vec3_scale(rikinv,delrik,rik_hat);

  // from PairTersoff::ters_zetaterm_d
  F_FLOAT gijk,dgijk,ex_delr,dex_delr,fc,dfc,cos,tmp;
  F_FLOAT dcosfi[3],dcosfj[3],dcosfk[3];

  fc = ters_fc_k(i,j,k,rik);
  dfc = ters_dfc(i,j,k,rik);
  if (int(paramskk(i,j,k).powerm) == 3) tmp = pow(paramskk(i,j,k).lam3 * (rij-rik),3.0);
  else tmp = paramskk(i,j,k).lam3 * (rij-rik);

  if (tmp > 69.0776) ex_delr = 1.e30;
  else if (tmp < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(tmp);

  if (int(paramskk(i,j,k).powerm) == 3)
    dex_delr = 3.0*pow(paramskk(i,j,k).lam3,3.0) * pow(rij-rik,2.0)*ex_delr;
  else dex_delr = paramskk(i,j,k).lam3 * ex_delr;

  cos = vec3_dot(rij_hat,rik_hat);
  gijk = ters_gijk(i,j,k,cos);
  dgijk = ters_dgijk(i,j,k,cos);

  // from PairTersoff::costheta_d
  vec3_scaleadd(-cos,rij_hat,rik_hat,dcosfj);
  vec3_scale(rijinv,dcosfj,dcosfj);
  vec3_scaleadd(-cos,rik_hat,rij_hat,dcosfk);
  vec3_scale(rikinv,dcosfk,dcosfk);
  vec3_add(dcosfj,dcosfk,dcosfi);
  vec3_scale(-1.0,dcosfi,dcosfi);

  vec3_scale(-dfc*gijk*ex_delr,rik_hat,fi);
  vec3_scaleadd(fc*dgijk*ex_delr,dcosfi,fi,fi);
  vec3_scaleadd(fc*gijk*dex_delr,rik_hat,fi,fi);
  vec3_scaleadd(-fc*gijk*dex_delr,rij_hat,fi,fi);
  vec3_scale(prefactor,fi,fi);

  vec3_scale(fc*dgijk*ex_delr,dcosfj,fj);
  vec3_scaleadd(fc*gijk*dex_delr,rij_hat,fj,fj);
  vec3_scale(prefactor,fj,fj);

  vec3_scale(dfc*gijk*ex_delr,rik_hat,fk);
  vec3_scaleadd(fc*dgijk*ex_delr,dcosfk,fk,fk);
  vec3_scaleadd(-fc*gijk*dex_delr,rik_hat,fk,fk);
  vec3_scale(prefactor,fk,fk);

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairTersoffKokkos<DeviceType>::ters_dthbj(
	const int &i, const int &j, const int &k, const F_FLOAT &prefactor,
	const F_FLOAT &rij, const F_FLOAT &dx1, const F_FLOAT &dy1, const F_FLOAT &dz1,
	const F_FLOAT &rik, const F_FLOAT &dx2, const F_FLOAT &dy2, const F_FLOAT &dz2,
	F_FLOAT *fj, F_FLOAT *fk) const
{
  F_FLOAT rij_hat[3],rik_hat[3];
  F_FLOAT rijinv,rikinv;
  F_FLOAT delrij[3], delrik[3];

  delrij[0] = dx1; delrij[1] = dy1; delrij[2] = dz1;
  delrik[0] = dx2; delrik[1] = dy2; delrik[2] = dz2;

  rijinv = 1.0/rij;
  vec3_scale(rijinv,delrij,rij_hat);

  rikinv = 1.0/rik;
  vec3_scale(rikinv,delrik,rik_hat);

  F_FLOAT gijk,dgijk,ex_delr,dex_delr,fc,dfc,cos,tmp;
  F_FLOAT dcosfi[3],dcosfj[3],dcosfk[3];

  fc = ters_fc_k(i,j,k,rik);
  dfc = ters_dfc(i,j,k,rik);
  if (int(paramskk(i,j,k).powerm) == 3) tmp = pow(paramskk(i,j,k).lam3 * (rij-rik),3.0);
  else tmp = paramskk(i,j,k).lam3 * (rij-rik);

  if (tmp > 69.0776) ex_delr = 1.e30;
  else if (tmp < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(tmp);

  if (int(paramskk(i,j,k).powerm) == 3)
    dex_delr = 3.0*pow(paramskk(i,j,k).lam3,3.0) * pow(rij-rik,2.0)*ex_delr;
  else dex_delr = paramskk(i,j,k).lam3 * ex_delr;

  cos = vec3_dot(rij_hat,rik_hat);
  gijk = ters_gijk(i,j,k,cos);
  dgijk = ters_dgijk(i,j,k,cos);

  vec3_scaleadd(-cos,rij_hat,rik_hat,dcosfj);
  vec3_scale(rijinv,dcosfj,dcosfj);
  vec3_scaleadd(-cos,rik_hat,rij_hat,dcosfk);
  vec3_scale(rikinv,dcosfk,dcosfk);
  vec3_add(dcosfj,dcosfk,dcosfi);
  vec3_scale(-1.0,dcosfi,dcosfi);

  vec3_scale(fc*dgijk*ex_delr,dcosfj,fj);
  vec3_scaleadd(fc*gijk*dex_delr,rij_hat,fj,fj);
  vec3_scale(prefactor,fj,fj);

  vec3_scale(dfc*gijk*ex_delr,rik_hat,fk);
  vec3_scaleadd(fc*dgijk*ex_delr,dcosfk,fk,fk);
  vec3_scaleadd(-fc*gijk*dex_delr,rik_hat,fk,fk);
  vec3_scale(prefactor,fk,fk);

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairTersoffKokkos<DeviceType>::ters_dthbk(
	const int &i, const int &j, const int &k, const F_FLOAT &prefactor,
	const F_FLOAT &rij, const F_FLOAT &dx1, const F_FLOAT &dy1, const F_FLOAT &dz1,
	const F_FLOAT &rik, const F_FLOAT &dx2, const F_FLOAT &dy2, const F_FLOAT &dz2,
	F_FLOAT *fk) const
{
  F_FLOAT rij_hat[3],rik_hat[3];
  F_FLOAT rijinv,rikinv;
  F_FLOAT delrij[3], delrik[3];

  delrij[0] = dx1; delrij[1] = dy1; delrij[2] = dz1;
  delrik[0] = dx2; delrik[1] = dy2; delrik[2] = dz2;

  rijinv = 1.0/rij;
  vec3_scale(rijinv,delrij,rij_hat);

  rikinv = 1.0/rik;
  vec3_scale(rikinv,delrik,rik_hat);

  F_FLOAT gijk,dgijk,ex_delr,dex_delr,fc,dfc,cos,tmp;
  F_FLOAT dcosfi[3],dcosfj[3],dcosfk[3];

  fc = ters_fc_k(i,j,k,rik);
  dfc = ters_dfc(i,j,k,rik);
  if (int(paramskk(i,j,k).powerm) == 3) tmp = pow(paramskk(i,j,k).lam3 * (rij-rik),3.0);
  else tmp = paramskk(i,j,k).lam3 * (rij-rik);

  if (tmp > 69.0776) ex_delr = 1.e30;
  else if (tmp < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(tmp);

  if (int(paramskk(i,j,k).powerm) == 3)
    dex_delr = 3.0*pow(paramskk(i,j,k).lam3,3.0) * pow(rij-rik,2.0)*ex_delr;
  else dex_delr = paramskk(i,j,k).lam3 * ex_delr;

  cos = vec3_dot(rij_hat,rik_hat);
  gijk = ters_gijk(i,j,k,cos);
  dgijk = ters_dgijk(i,j,k,cos);

  vec3_scaleadd(-cos,rij_hat,rik_hat,dcosfj);
  vec3_scale(rijinv,dcosfj,dcosfj);
  vec3_scaleadd(-cos,rik_hat,rij_hat,dcosfk);
  vec3_scale(rikinv,dcosfk,dcosfk);
  vec3_add(dcosfj,dcosfk,dcosfi);
  vec3_scale(-1.0,dcosfi,dcosfi);

  vec3_scale(dfc*gijk*ex_delr,rik_hat,fk);
  vec3_scaleadd(fc*dgijk*ex_delr,dcosfk,fk,fk);
  vec3_scaleadd(-fc*gijk*dex_delr,rik_hat,fk,fk);
  vec3_scale(prefactor,fk,fk);

}
/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairTersoffKokkos<DeviceType>::v_tally3_atom(EV_FLOAT &ev, const int &i, const int &j, const int &k,
        F_FLOAT *fj, F_FLOAT *fk, F_FLOAT *drji, F_FLOAT *drjk) const
{
  F_FLOAT v[6];

  v[0] = THIRD * (drji[0]*fj[0] + drjk[0]*fk[0]);
  v[1] = THIRD * (drji[1]*fj[1] + drjk[1]*fk[1]);
  v[2] = THIRD * (drji[2]*fj[2] + drjk[2]*fk[2]);
  v[3] = THIRD * (drji[0]*fj[1] + drjk[0]*fk[1]);
  v[4] = THIRD * (drji[0]*fj[2] + drjk[0]*fk[2]);
  v[5] = THIRD * (drji[1]*fj[2] + drjk[1]*fk[2]);

  if (vflag_global) {
    ev.v[0] += v[0];
    ev.v[1] += v[1];
    ev.v[2] += v[2];
    ev.v[3] += v[3];
    ev.v[4] += v[4];
    ev.v[5] += v[5];
  }

  if (vflag_atom) {
    d_vatom(i,0) += v[0]; d_vatom(i,1) += v[1]; d_vatom(i,2) += v[2];
    d_vatom(i,3) += v[3]; d_vatom(i,4) += v[4]; d_vatom(i,5) += v[5];
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int PairTersoffKokkos<DeviceType>::sbmask(const int& j) const {
  return j >> SBBITS & 3;
}

// The factor up to which we do caching
static const int N_CACHE = 8;

template<class DeviceType>
template<int EVFLAG, int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairTersoffKokkos<DeviceType>::kernel_step(
    int compress_idx, 
    iarr is,
    iarr js,
    bvec vmask_repulsive,
    EV_FLOAT &ev
) const {

  Kokkos::View<F_FLOAT*[3], typename DAT::t_f_array::array_layout,DeviceType,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > a_f = f;
  ivec v_i4floats((int) (4 * sizeof(typename v::fscal)));
  ivec v_i1(1);
  fvec v_2(0.);
  fvec v_0_5(0.5);
  ivec v_i0(0);
  ivec v_i_ntypes(ntypes);
  ivec v_i_NEIGHMASK(NEIGHMASK);
  
  farr fx, fy, fz, fw;
  int cache_idx = 0;
  fvec vfkx_cache[N_CACHE+1];
  fvec vfky_cache[N_CACHE+1];
  fvec vfkz_cache[N_CACHE+1];
  ivec vks_cache[N_CACHE+1];
  bvec vmask_cache[N_CACHE+1];
  ivec vkks_final_cache;
  bvec vmask_final_cache;
  iarr ts; 
  // compute all the stuff we know from i and j
  // TDO: We could extract this from the driver routine
  ivec vis = v::int_mullo(v_i4floats, v::int_load_vl(is));
  ivec vjs = v::int_mullo(v_i4floats, v::int_load_vl(js));
  bvec vmask = v::mask_enable_lower(compress_idx);
  fvec vx_i = v::zero(), vy_i = v::zero(), vz_i = v::zero();
  ivec vw_i = v_i0;
  //v::gather_x(vis, vmask, x, &vx_i, &vy_i, &vz_i, &vw_i);
  vx_i = x(is[0], 0);
  vy_i = x(is[0], 1);
  vz_i = x(is[0], 2);
  vw_i = type(is[0]);
  fvec vx_j = v::zero(), vy_j = v::zero(), vz_j = v::zero();
  ivec vw_j = v_i0;
  //v::gather_x(vjs, vmask, x, &vx_j, &vy_j, &vz_j, &vw_j);
  vx_j = x(js[0], 0);
  vy_j = x(js[0], 1);
  vz_j = x(js[0], 2);
  vw_j = type(js[0]);
  fvec vdx_ij = vx_j - vx_i, vdy_ij = vy_j - vy_i, vdz_ij = vz_j - vz_i;
  fvec vrijsq = vdx_ij * vdx_ij + vdy_ij *  vdy_ij + vdz_ij * vdz_ij;
  fvec vrij = sqrt(vrijsq);
  ivec vis_orig = v::int_load_vl(is);
  //ivec vcnumneigh_i = v::int_gather<4>(v_i0, vmask, vis_orig, cnumneigh);
  ivec vnumneigh_i = d_numneigh[vis_orig];//v::int_gather<4>(v_i0, vmask, vis_orig, numneigh);
  ivec vc_idx_ij = v::int_mullo(v_i4floats, vw_j + v::int_mullo(v_i_ntypes, vw_i));

  fvec vzeta = v::zero();
  fvec vfxtmp = v::zero(), vfytmp = v::zero(), vfztmp = v::zero();
  fvec vfjxtmp = v::zero(), vfjytmp = v::zero(), vfjztmp = v::zero();
  // This piece of code faciliates the traversal of the k loop assuming
  //  nothing about i. As such, it uses masking to avoid superfluous loads
  //  and fast-forwards each lane until work is available.
  // This is useful because we can not make assumptions as to where in the
  //  neighbor list the atoms within the cutoff might be.
  // We also implement the caching in here, i.e. collect force contributions
  //  due to zeta.
  // This means that you will see four loops:
  // 1. the loop that does zeta calculation and caches the force contributions
  // 2. the loop that processes the remaining zeta calculations
  // 3. the loop that updates the force based on the cached force contributions
  // 4. the loop that computes force contributions for the remainder
  {
    ivec vkks = v_i0;
    bvec vactive_mask = vmask & v::int_cmplt(vkks, vnumneigh_i);;
    bvec veff_old_mask(0);
    ivec vks, vw_k;
    fvec vx_k, vy_k, vz_k, vcutsq;
    while (! v::mask_testz(vactive_mask) && cache_idx < N_CACHE) {
      bvec vnew_mask = vactive_mask & ~ veff_old_mask;
      if(vactive_mask) vks = d_neighbors(vis_orig, vkks);
//      vks = v::int_mullo(v_i4floats, v_i_NEIGHMASK &
//          v::int_gather<4>(vks, vactive_mask, vkks + vcnumneigh_i, firstneigh));
      //v::gather_x(vks, vnew_mask, x, &vx_k, &vy_k, &vz_k, &vw_k);
      vx_k = x(vks, 0);
      vy_k = x(vks, 1);
      vz_k = x(vks, 2);
      vw_k = type(vks);
      fvec vdx_ik = (vx_k - vx_i);
      fvec vdy_ik = (vy_k - vy_i);
      fvec vdz_ik = (vz_k - vz_i);
      fvec vrsq = vdx_ik * vdx_ik + vdy_ik *  vdy_ik + vdz_ik * vdz_ik;
      ivec vc_idx = v::int_mullo(v_i4floats, vw_k) + v::int_mullo(v_i_ntypes, vc_idx_ij);
      //vcutsq = v::gather<4>(vcutsq, vnew_mask, vc_idx, c_inner);
      vcutsq = paramskk(vw_i, vw_j, vw_k).cutsq;
      bvec vcutoff_mask = v::cmplt(vrsq, vcutsq);
      bvec vsame_mask = v::int_cmpneq(vjs, 4 * sizeof(typename v::fscal) * vks);
      bvec veff_mask = vcutoff_mask & vsame_mask & vactive_mask;
      if (v::mask_testz(!(veff_mask || !vactive_mask))) {
        fvec vzeta_contrib;
        fvec vfix, vfiy, vfiz;
        fvec vfjx, vfjy, vfjz;
        fvec vfkx, vfky, vfkz;

        attractive_vector<true>(vc_idx,veff_mask,fvec(1.),
            vrij,vrsq,vdx_ij,vdy_ij,vdz_ij,vdx_ik,vdy_ik,vdz_ik,
            &vfix,&vfiy,&vfiz,
            &vfjx,&vfjy,&vfjz,
            &vfkx,&vfky,&vfkz,
	    &vzeta_contrib);
        vfxtmp = v::mask_add(vfxtmp, veff_mask, vfxtmp, vfix);
        vfytmp = v::mask_add(vfytmp, veff_mask, vfytmp, vfiy);
        vfztmp = v::mask_add(vfztmp, veff_mask, vfztmp, vfiz);
        vfjxtmp = v::mask_add(vfjxtmp, veff_mask, vfjxtmp, vfjx);
        vfjytmp = v::mask_add(vfjytmp, veff_mask, vfjytmp, vfjy);
        vfjztmp = v::mask_add(vfjztmp, veff_mask, vfjztmp, vfjz);

        vfkx_cache[cache_idx] = vfkx;
        vfky_cache[cache_idx] = vfky;
        vfkz_cache[cache_idx] = vfkz;
	vks_cache[cache_idx] = vks;
	vmask_cache[cache_idx] = veff_mask;
	cache_idx += 1;

        vzeta = v::mask_add(vzeta, veff_mask, vzeta, vzeta_contrib);
        vkks = vkks + v_i1;
        veff_old_mask = bvec(0);
      } else {
        vkks = v::int_mask_add(vkks, !veff_mask, vkks, v_i1);
        veff_old_mask = veff_mask;
      }
      vactive_mask &= v::int_cmplt(vkks, vnumneigh_i);
    }
    vkks_final_cache = vkks;
    vmask_final_cache = vactive_mask;
    while (! v::mask_testz(vactive_mask)) {
      bvec vnew_mask = vactive_mask & ~ veff_old_mask;
      if(vactive_mask) vks = d_neighbors(vis_orig, vkks);
      vx_k = x(vks, 0);
      vy_k = x(vks, 1);
      vz_k = x(vks, 2);
      vw_k = type(vks);

     // vks = v::int_mullo(v_i4floats, v_i_NEIGHMASK &
     //     v::int_gather<4>(vks, vactive_mask, vkks + vcnumneigh_i, firstneigh));
     // v::gather_x(vks, vnew_mask, x, &vx_k, &vy_k, &vz_k, &vw_k);
      fvec vdx_ik = (vx_k - vx_i);
      fvec vdy_ik = (vy_k - vy_i);
      fvec vdz_ik = (vz_k - vz_i);
      fvec vrsq = vdx_ik * vdx_ik + vdy_ik *  vdy_ik + vdz_ik * vdz_ik;
      ivec vc_idx = v::int_mullo(v_i4floats, vw_k) + v::int_mullo(v_i_ntypes, vc_idx_ij);
      //vcutsq = v::gather<4>(vcutsq, vnew_mask, vc_idx, c_inner);
      vcutsq = paramskk(vw_i, vw_j, vw_k).cutsq;
      bvec vcutoff_mask = v::cmplt(vrsq, vcutsq);
      bvec vsame_mask = v::int_cmpneq(vjs, 4 * sizeof(typename v::fscal) * vks);
      bvec veff_mask = vcutoff_mask & vsame_mask & vactive_mask;
      if (v::mask_testz(!(veff_mask || !vactive_mask))) {
        fvec vzeta_contrib;
        vzeta_contrib = zeta_vector(vc_idx,veff_mask,vrij,vrsq,vdx_ij,vdy_ij,vdz_ij,vdx_ik,vdy_ik,vdz_ik);
        vzeta = v::mask_add(vzeta, veff_mask, vzeta, vzeta_contrib);
        vkks = vkks + v_i1;
        veff_old_mask = bvec(0);
      } else {
        vkks = v::int_mask_add(vkks, !veff_mask, vkks, v_i1);
        veff_old_mask = veff_mask;
      }
      vactive_mask &= v::int_cmplt(vkks, vnumneigh_i);
    }
  }
  fvec vfpair, vevdwl, vprefactor, vfwtmp, vfjwtmp;
  force_zeta_vector(vc_idx_ij, vmask, vrij, vzeta, &vfpair, &vprefactor, eflag, &vevdwl, vmask_repulsive);
//printf("%d %d fzeta %e %e\n", is[0], js[0], vfpair, vprefactor);
  vfxtmp = vfxtmp * vprefactor + vdx_ij * vfpair;
  vfytmp = vfytmp * vprefactor + vdy_ij * vfpair;
  vfztmp = vfztmp * vprefactor + vdz_ij * vfpair;
  vfjxtmp = vfjxtmp * vprefactor - vdx_ij * vfpair;
  vfjytmp = vfjytmp * vprefactor - vdy_ij * vfpair;
  vfjztmp = vfjztmp * vprefactor - vdz_ij * vfpair;
 
  if (EVFLAG) {
    if (eflag) ev.evdwl += vevdwl;
    if (vflag_either || eflag_atom)
      this->template ev_tally<NEIGHFLAG>(ev,vis,vjs,vevdwl,vfpair,vdx_ij,vdy_ij,vdz_ij);
  }
  {
    while (cache_idx-- > 0) {
      fvec vfkx = vprefactor * vfkx_cache[cache_idx];
      fvec vfky = vprefactor * vfky_cache[cache_idx];
      fvec vfkz = vprefactor * vfkz_cache[cache_idx];
      ivec vks = vks_cache[cache_idx];
      bvec veff_mask = vmask_cache[cache_idx];
      if (veff_mask) {
        int k = vks; // (4*sizeof(typename v::fscal));
        a_f(k, 0) += vfkx;
        a_f(k, 1) += vfky;
        a_f(k, 2) += vfkz;
      }
      //v::store(fx, vfkx);
      //v::store(fy, vfky);
      //v::store(fz, vfkz);
      //v::int_store(ts, vks);
      //for (int t = 0; t < v::VL; t++) {
      //  if (v::mask_test_at(veff_mask, t)) {
      //    int t_ = ts[t] / (4 * sizeof(typename v::fscal));
      //    f[t_].x += fx[t];
      //    f[t_].y += fy[t];
      //    f[t_].z += fz[t];
      //  }
      //}
    }
    ivec vkks = vkks_final_cache;
    bvec vactive_mask = vmask_final_cache;
    bvec veff_old_mask(0);
    ivec vks, vw_k;
    fvec vx_k, vy_k, vz_k, vcutsq;
    while (! v::mask_testz(vactive_mask)) {
      bvec vnew_mask = vactive_mask & ~ veff_old_mask;
      if(vactive_mask) vks = d_neighbors(vis_orig, vkks);
      vx_k = x(vks, 0);
      vy_k = x(vks, 1);
      vz_k = x(vks, 2);
      vw_k = type(vks);
      //vks = v::int_mullo(v_i4floats, v_i_NEIGHMASK & 
      //    v::int_gather<4>(vks, vactive_mask, vkks + vcnumneigh_i, firstneigh));
      //v::gather_x(vks, vnew_mask, x, &vx_k, &vy_k, &vz_k, &vw_k);
      fvec vdx_ik = vx_k - vx_i;
      fvec vdy_ik = vy_k - vy_i;
      fvec vdz_ik = vz_k - vz_i;
      fvec vrsq = vdx_ik * vdx_ik + vdy_ik *  vdy_ik + vdz_ik * vdz_ik;
      ivec vc_idx = v::int_mullo(v_i4floats, vw_k) + v::int_mullo(v_i_ntypes, vc_idx_ij);
      //vcutsq = v::gather<4>(vcutsq, vnew_mask, vc_idx, c_inner);
      vcutsq = paramskk(vw_i, vw_j, vw_k).cutsq;
      bvec vcutoff_mask = v::cmplt(vrsq, vcutsq);
      bvec vsame_mask = v::int_cmpneq(vjs, 4 * sizeof(typename v::fscal) * vks);
      bvec veff_mask = vcutoff_mask & vsame_mask & vactive_mask;
      if (v::mask_testz(!(veff_mask || !vactive_mask))) {
        fvec vfix, vfiy, vfiz;
        fvec vfjx, vfjy, vfjz;
        fvec vfkx, vfky, vfkz;

        attractive_vector<false>(vc_idx,veff_mask,vprefactor,
            vrij,vrsq,vdx_ij,vdy_ij,vdz_ij,vdx_ik,vdy_ik,vdz_ik,
            &vfix,&vfiy,&vfiz,
            &vfjx,&vfjy,&vfjz,
            &vfkx,&vfky,&vfkz,
	    0);
//printf("ATT %e %e %e %e %e %e %e %e %e\n", vfix, vfiy, vfiz, vfjx, vfjy, vfjz, vfkx, vfky, vfkz);
        vfxtmp = v::mask_add(vfxtmp, veff_mask, vfxtmp, vfix);
        vfytmp = v::mask_add(vfytmp, veff_mask, vfytmp, vfiy);
        vfztmp = v::mask_add(vfztmp, veff_mask, vfztmp, vfiz);
        vfjxtmp = v::mask_add(vfjxtmp, veff_mask, vfjxtmp, vfjx);
        vfjytmp = v::mask_add(vfjytmp, veff_mask, vfjytmp, vfjy);
        vfjztmp = v::mask_add(vfjztmp, veff_mask, vfjztmp, vfjz);
        if (veff_mask) {
          int k = vks;// / (4*sizeof(typename v::fscal));
          a_f(k, 0) += vfkx;
          a_f(k, 1) += vfky;
          a_f(k, 2) += vfkz;
        }

        //v::store(fx, vfkx);
        //v::store(fy, vfky);
        //v::store(fz, vfkz);
        //v::int_store(ts, vks);
        //for (int t = 0; t < v::VL; t++) {
        //  if (v::mask_test_at(veff_mask, t)) {
        //    int t_ = ts[t] / (4 * sizeof(typename v::fscal));
        //    f[t_].x += fx[t];
        //    f[t_].y += fy[t];
        //    f[t_].z += fz[t];
        //  }
        //}
        vkks = vkks + v_i1;
        veff_old_mask = bvec(0);
      } else {
        vkks = v::int_mask_add(vkks, !veff_mask, vkks, v_i1);
        veff_old_mask = veff_mask;
      }
      vactive_mask &= v::int_cmplt(vkks, vnumneigh_i);
    } // while (vactive_mask != 0)
  } // section k
  // We can not make any assumptions regarding conflicts.
  // So we sequentialize this.
  // TDO: Once AVX-512 is around check out VPCONFLICT
  a_f(is[0], 0) += vfxtmp;
  a_f(is[0], 1) += vfytmp;
  a_f(is[0], 2) += vfztmp;

  a_f(js[0], 0) += vfjxtmp;
  a_f(js[0], 1) += vfjytmp;
  a_f(js[0], 2) += vfjztmp;
  //v::store(fx, vfjxtmp);
  //v::store(fy, vfjytmp);
  //v::store(fz, vfjztmp);
  //for (int t = 0; t < compress_idx; t++) {
  //  int t_ = js[t];
  //  f[t_].x += fx[t];
  //  f[t_].y += fy[t];
  //  f[t_].z += fz[t];
  //  if (EVFLAG && eflag && eatom) {
  //    f[t_].w += fw[t];
  //  }
  //}
  //v::store(fx, vfxtmp);
  //v::store(fy, vfytmp);
  //v::store(fz, vfztmp);
  //for (int t = 0; t < compress_idx; t++) {
  //  int t_ = is[t];
  //  f[t_].x += fx[t];
  //  f[t_].y += fy[t];
  //  f[t_].z += fz[t];
  //  if (EVFLAG && eflag && eatom) {
  //    f[t_].w += fw[t];
  //  }
  //}
}

template<class DeviceType>
typename PairTersoffKokkos<DeviceType>::fvec PairTersoffKokkos<DeviceType>::zeta_vector(
    ivec xjw, bvec mask, 
    fvec vrij, fvec rsq2, 
    fvec vdijx, fvec vdijy, fvec vdijz, 
    fvec dikx, fvec diky, fvec dikz
) const {
  fvec v_1_0(1.0);
  fvec v_0_5(0.5);
  fvec vph = v::zero(), vpc2 = v::zero(), vpd2 = v::zero(), vpgamma = v::zero(), vplam3 = v::zero(), vppowermint = v::zero(), vpbigr = v::zero(), vpbigd = v::zero();

  int tp1 = ntypes;
  int rest1 = xjw / 4 / sizeof(typename v::fscal);
  int w_k = (rest1-1) % tp1 + 1;
  int rest2 = (rest1-w_k) / tp1;
  int w_j = (rest2-1) % tp1 + 1;
  int w_i = (rest2-w_j) / tp1;

  // TDO: Specialize on number of species
  //v::gather_8(xjw, mask, &param[0].lam3, &vplam3, &vppowermint, &vpbigr, &vpbigd, &vpc2, &vpd2, &vph, &vpgamma);
  vplam3 = paramskk(w_i, w_j, w_k).lam3;
  vpbigr = paramskk(w_i, w_j, w_k).bigr;
  vpbigd = paramskk(w_i, w_j, w_k).bigd;
  vpc2 = paramskk(w_i, w_j, w_k).c; vpc2 *= vpc2;
  vpd2 = paramskk(w_i, w_j, w_k).d; vpd2 *= vpd2;
  vpgamma = paramskk(w_i, w_j, w_k).gamma;
  vppowermint = paramskk(w_i, w_j, w_k).powerm;
  vph = paramskk(w_i, w_j, w_k).h;
  fvec vrik = sqrt(rsq2);
  fvec vcostheta = (vdijx * dikx + vdijy * diky + vdijz * dikz) * v::recip(vrij * vrik);
  fvec vhcth = vph - vcostheta;
  fvec vgijk_a = vhcth * vhcth;
  fvec vgijk = vpgamma * (v_1_0 + vpc2 * vgijk_a * v::recip(vpd2 * (vpd2 + vgijk_a)));
  fvec varg1 = vplam3 * (vrij - vrik);
  fvec varg3 = varg1 * varg1 * varg1;
  bvec mask_ex = v::cmpeq(vppowermint, fvec(3.));
  fvec varg  = v::blend(mask_ex, varg1, varg3);
  fvec vex_delr = v::min(fvec(1.e30), exp(varg));
  bvec vmask_need_sine = v::cmpnle(vrik, vpbigr - vpbigd) & mask;
  fvec vfc = v_1_0;
  // Its kind of important to check the mask.
  // Some simulations never/rarely invoke this branch.
  if (! v::mask_testz(vmask_need_sine)) {
    vfc = v::blend(vmask_need_sine, vfc, 
        v_0_5 * (v_1_0 - sin(fvec(MY_PI2) * (vrik - vpbigr) * v::recip(vpbigd))));
  }
  return vgijk * vex_delr * vfc;
}

template<class DeviceType>
void PairTersoffKokkos<DeviceType>::force_zeta_vector(
    ivec xjw,
    bvec mask,
    fvec vrij, fvec vzeta_ij,
    fvec *vfpair, fvec *vprefactor, int EVDWL, fvec *vevdwl,
    bvec vmask_repulsive
) const {
  fvec v_0_0(0.0);
  fvec v_0_5(0.5);
  fvec v_m0_5(-0.5);
  fvec v_1_0(1.0);
  fvec v_m1_0(-1.0);
  fvec v_2_0(2.0);
  fvec vpbigr = v::zero(), vpbigd = v::zero(), vplam1 = v::zero(), vpbiga = v::zero(), vplam2 = v::zero(), vpbeta = v::zero(), vpbigb = v::zero(), vppowern = v::zero();
  int tp1 = ntypes;
  int rest1 = xjw / 4 / sizeof(typename v::fscal);
  int w_j = (rest1-1) % tp1 + 1;
  int rest2 = (rest1-w_j) / tp1;
  int w_i = (rest2-1) % tp1 + 1;
  // TDO: Specialize on number of species
  vpbigr = paramskk(w_i, w_j, w_j).bigr;
  vpbigd = paramskk(w_i, w_j, w_j).bigd;
  vplam1 = paramskk(w_i, w_j, w_j).lam1;
  vpbiga = paramskk(w_i, w_j, w_j).biga;
  vplam2 = paramskk(w_i, w_j, w_j).lam2;
  vpbeta = paramskk(w_i, w_j, w_j).beta;
  vpbigb = paramskk(w_i, w_j, w_j).bigb;
  vppowern = paramskk(w_i, w_j, w_j).powern;
 
  //v::gather_8(xjw, mask, &param[0].bigr, &vpbigr, &vpbigd, &vplam1, &vpbiga, &vplam2, &vpbeta, &vpbigb, &vppowern);
  fvec vfccos;

  // This is pretty much a literal translation.
  bvec vmask_need_sine = v::cmpnle(vrij, vpbigr - vpbigd) & mask;
  fvec vfc = v_1_0;
  fvec vfc_d = v_0_0;
  if (! v::mask_testz(vmask_need_sine)) {
    fvec vtmp = fvec(MY_PI2) * v::recip(vpbigd);
    vfc = v::blend(vmask_need_sine, vfc,
        v_0_5 * (v_1_0 - v::sincos(&vfccos, vtmp * (vrij - vpbigr))));
    vfc_d = v::blend(vmask_need_sine, vfc_d, v_m0_5 * vtmp * vfccos);
  }
  fvec vpminus_lam2 =  - vplam2;

  fvec vpminus_bigb = -vpbigb;
  fvec vexp = exp(vpminus_lam2 * vrij);
  fvec vfa = vpminus_bigb * vexp * vfc;
  fvec vfa_d = vpminus_lam2 * vfa + vpminus_bigb * vexp * vfc_d;

  fvec vpc1 = v::zero(), vpc2 = v::zero(), vpc3 = v::zero(), vpc4 = v::zero();
  //v::gather_4(xjw, mask, &param[0].c1, &vpc1, &vpc2, &vpc3, &vpc4);
  vpc1 = paramskk(w_i, w_j, w_j).c1;
  vpc2 = paramskk(w_i, w_j, w_j).c2;
  vpc3 = paramskk(w_i, w_j, w_j).c3;
  vpc4 = paramskk(w_i, w_j, w_j).c4;
  fvec vpminus_powern = - vppowern;
  fvec vbij(0.), vbij_d(0.);
  fvec vtmp = vpbeta * vzeta_ij;
  bvec vmc1 = v::cmple(vpc1, vtmp) & mask;
  if (! v::mask_testz(vmc1)) {
    vbij = v::invsqrt(vtmp);
    vbij_d = vpbeta * v_m0_5 * vbij * v::recip(vtmp);
//printf("A\n");
  }
  bvec vmc2 = v::cmple(vpc2, vtmp) & ~ vmc1 & mask;
  if (! v::mask_testz(vmc2)) {
    fvec vpowminus_powern = pow(vtmp, vpminus_powern);
    fvec vinvsqrt = v::invsqrt(vtmp);
    fvec vrcp2powern = v::recip(v_2_0 * vppowern);
    fvec va = (v_1_0 - vpowminus_powern * vrcp2powern) * vinvsqrt;
    fvec va_d = vpbeta * v_m0_5 * vinvsqrt * v::recip(vtmp) *
            (v_1_0 + v_m0_5 * vpowminus_powern * (v_1_0 + vrcp2powern));
    vbij = v::blend(vmc2, vbij, va);
    vbij_d = v::blend(vmc2, vbij_d, va_d);
//printf("B\n");
  }
  bvec vmc3 = v::cmplt(vtmp, vpc4) & ~vmc2 & ~vmc1 & mask;
  if (! v::mask_testz(vmc3)) {
    vbij = v::blend(vmc3, vbij, v_1_0);
    vbij_d = v::blend(vmc3, vbij_d, v_0_0);
//printf("C\n");
  }
  bvec vmc4 = v::cmple(vtmp, vpc3) & ~vmc3 & ~vmc2 & ~ vmc1 & mask;
  if (! v::mask_testz(vmc4)) {
    fvec vpowm1 = pow(vtmp, vppowern - v_1_0);
    fvec vrcp2powern = v::recip(v_2_0 * vppowern);
    fvec va = v_1_0 - vtmp * vrcp2powern * vpowm1;
    fvec va_d = v_m0_5 * vpbeta * vpowm1;
    vbij = v::blend(vmc4, vbij, va);
    vbij_d = v::blend(vmc4, vbij_d, va_d);
//printf("D %e %e", vbij, vbij_d);
  }
  bvec vmc5 = mask & ~vmc1 & ~vmc2 & ~vmc3 & ~vmc4;
  if (! v::mask_testz(vmc5)) {
    fvec vtmp_n = pow(vtmp, vppowern);
    fvec vpow2 = pow(v_1_0 + vtmp_n, v_m1_0 - v::recip(v_2_0 * vppowern));
    fvec va = (v_1_0 + vtmp_n) * vpow2;
    fvec va_d = v_m0_5 * vpow2 * vtmp_n * v::recip(vzeta_ij);
    vbij = v::blend(vmc5, vbij, va);
    vbij_d = v::blend(vmc5, vbij_d, va_d);
//printf("E");
  }
  fvec vtmp_exp = exp(-vplam1 * vrij);
  fvec vrep_fforce = vpbiga * vtmp_exp * (vfc_d - vfc * vplam1);
  fvec vfz_fforce = v_0_5 * vbij * vfa_d;
//printf(" %e %e\n", vfa, vfa_d);

  *vfpair = v::mask_add(vfz_fforce, vmask_repulsive, vfz_fforce, vrep_fforce) * v::recip(vrij);
//printf("att %e\n", vfz_fforce/vrij);
  *vprefactor = v_m0_5 * vfa * vbij_d;
  if (EVDWL) {
    fvec vrep_eng = vfc * vpbiga * vtmp_exp;
    fvec vfz_eng = v_0_5 * vfa * vbij;
    *vevdwl = v::mask_add(vfz_eng, vmask_repulsive, vfz_eng, vrep_eng);
  }
}

template<class DeviceType>
template<bool ZETA>
void PairTersoffKokkos<DeviceType>::attractive_vector(
    ivec xjw,
    bvec mask,
    fvec vprefactor,
    fvec vrij, fvec rsq2,
    fvec vdijx, fvec vdijy, fvec vdijz,
    fvec dikx, fvec diky, fvec dikz,
    fvec *fix, fvec *fiy, fvec *fiz,
    fvec *fjx, fvec *fjy, fvec *fjz,
    fvec *fkx, fvec *fky, fvec *fkz,
    fvec *zeta
) const {
  fvec v_1_0 = fvec(1.0);

  fvec vph = v::zero(), vpc2 = v::zero(), vpd2 = fvec(1.0), vpgamma = v::zero(), vplam3 = v::zero(), vppowermint = v::zero(), vpbigr = v::zero(), vpbigd = fvec(1.0);
  //v::gather_8(xjw, mask, &param[0].lam3, &vplam3, &vppowermint, &vpbigr, &vpbigd, &vpc2, &vpd2, &vph, &vpgamma);

  int tp1 = ntypes;
  int rest1 = xjw / 4 / sizeof(typename v::fscal);
  int w_k = (rest1-1) % tp1 + 1;
  int rest2 = (rest1-w_k) / tp1;
  int w_j = (rest2-1) % tp1 + 1;
  int w_i = (rest2-w_j) / tp1;
  // TDO: Specialize on number of species
  // TDO: Specialize on number of species
  //v::gather_8(xjw, mask, &param[0].lam3, &vplam3, &vppowermint, &vpbigr, &vpbigd, &vpc2, &vpd2, &vph, &vpgamma);
  vplam3 = paramskk(w_i, w_j, w_k).lam3;
  vpbigr = paramskk(w_i, w_j, w_k).bigr;
  vpbigd = paramskk(w_i, w_j, w_k).bigd;
  vpc2 = paramskk(w_i, w_j, w_k).c; vpc2 *= vpc2;
  vpd2 = paramskk(w_i, w_j, w_k).d; vpd2 *= vpd2;
  vpgamma = paramskk(w_i, w_j, w_k).gamma;
  vppowermint = paramskk(w_i, w_j, w_k).powerm;
  vph = paramskk(w_i, w_j, w_k).h;
 
  fvec vrijinv = v::recip(vrij);
  fvec vrij_hatx = vrijinv * vdijx;
  fvec vrij_haty = vrijinv * vdijy;
  fvec vrij_hatz = vrijinv * vdijz;
  fvec rikinv = invsqrt(rsq2);
  fvec rik_hatx = rikinv * dikx;
  fvec rik_haty = rikinv * diky;
  fvec rik_hatz = rikinv * dikz;

  fvec vrik = sqrt(rsq2);
  fvec vcostheta = (vdijx * dikx + vdijy * diky + vdijz * dikz) * v::recip(vrij * vrik);
  fvec vhcth = vph - vcostheta;
  fvec vdenominator = v::recip(vpd2 + vhcth * vhcth);
  fvec vgijk = vpgamma * (v_1_0 + vpc2 * v::recip(vpd2) - vpc2 * vdenominator);
  fvec vnumerator = fvec(-2.) * vpc2 * vhcth;
  fvec vgijk_d = vpgamma * vnumerator * vdenominator * vdenominator;
  fvec varg1 = vplam3 * (vrij - vrik);
  fvec varg3 = varg1 * varg1 * varg1;
  bvec mask_ex = v::cmpeq(vppowermint, fvec(3.));
  fvec varg  = v::blend(mask_ex, varg1, varg3);
  fvec vex_delr = min(fvec(1.e30), exp(varg));
  fvec vex_delr_d_factor = v::blend(mask, v_1_0, fvec(3.0) * varg1 * varg1);
  fvec vex_delr_d = vplam3 * vex_delr_d_factor * vex_delr;
  bvec vmask_need_sine = v::cmpnle(vrik, vpbigr - vpbigd) & mask;
  fvec vfccos;
  fvec vfc = v_1_0;
  fvec vfc_d = v::zero();
  if (! v::mask_testz(vmask_need_sine)) {
    fvec vtmp = fvec(MY_PI2) * v::recip(vpbigd);
    vfc = v::blend(vmask_need_sine, vfc,
        fvec(0.5) * (v_1_0 - v::sincos(&vfccos, vtmp * (vrik - vpbigr))));
    vfc_d = v::blend(vmask_need_sine, vfc_d, fvec(-0.5) * vtmp * vfccos);
  }

  fvec vzeta_d_fc = vfc_d * vgijk * vex_delr; 
  fvec vzeta_d_gijk = vfc * vgijk_d * vex_delr; 
  fvec vzeta_d_ex_delr = vfc * vgijk * vex_delr_d; 
  if (ZETA) *zeta = vfc * vgijk * vex_delr;

  fvec vminus_costheta = - vcostheta;
  fvec vdcosdrjx = vrijinv * fmadd(vminus_costheta, vrij_hatx, rik_hatx);
  fvec vdcosdrjy = vrijinv * fmadd(vminus_costheta, vrij_haty, rik_haty);
  fvec vdcosdrjz = vrijinv * fmadd(vminus_costheta, vrij_hatz, rik_hatz);
  fvec vdcosdrkx = rikinv * fmadd(vminus_costheta, rik_hatx, vrij_hatx);
  fvec vdcosdrky = rikinv * fmadd(vminus_costheta, rik_haty, vrij_haty);
  fvec vdcosdrkz = rikinv * fmadd(vminus_costheta, rik_hatz, vrij_hatz);
  fvec vdcosdrix = -(vdcosdrjx + vdcosdrkx);
  fvec vdcosdriy = -(vdcosdrjy + vdcosdrky);
  fvec vdcosdriz = -(vdcosdrjz + vdcosdrkz);
  
  *fix = vprefactor * (vzeta_d_gijk * vdcosdrix + vzeta_d_ex_delr * (rik_hatx - vrij_hatx) - vzeta_d_fc * rik_hatx);
  *fiy = vprefactor * (vzeta_d_gijk * vdcosdriy + vzeta_d_ex_delr * (rik_haty - vrij_haty) - vzeta_d_fc * rik_haty);
  *fiz = vprefactor * (vzeta_d_gijk * vdcosdriz + vzeta_d_ex_delr * (rik_hatz - vrij_hatz) - vzeta_d_fc * rik_hatz);
  *fjx = vprefactor * (vzeta_d_gijk * vdcosdrjx + vzeta_d_ex_delr * vrij_hatx);
  *fjy = vprefactor * (vzeta_d_gijk * vdcosdrjy + vzeta_d_ex_delr * vrij_haty);
  *fjz = vprefactor * (vzeta_d_gijk * vdcosdrjz + vzeta_d_ex_delr * vrij_hatz);
  *fkx = vprefactor * ((vzeta_d_fc - vzeta_d_ex_delr) * rik_hatx + vzeta_d_gijk * vdcosdrkx);
  *fky = vprefactor * ((vzeta_d_fc - vzeta_d_ex_delr) * rik_haty + vzeta_d_gijk * vdcosdrky);
  *fkz = vprefactor * ((vzeta_d_fc - vzeta_d_ex_delr) * rik_hatz + vzeta_d_gijk * vdcosdrkz);
}


template class PairTersoffKokkos<LMPDeviceType>;
#ifdef KOKKOS_HAVE_CUDA
template class PairTersoffKokkos<LMPHostType>;
#endif
