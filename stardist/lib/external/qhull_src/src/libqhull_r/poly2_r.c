/*<html><pre>  -<a                             href="qh-poly_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   poly2_r.c
   implements polygons and simplicies

   see qh-poly_r.htm, poly_r.h and libqhull_r.h

   frequently used code is in poly_r.c

   Copyright (c) 1993-2018 The Geometry Center.
   $Id: //main/2015/qhull/src/libqhull_r/poly2_r.c#39 $$Change: 2552 $
   $DateTime: 2018/12/29 15:39:43 $$Author: bbarber $
*/

#include "qhull_ra.h"

/*======== functions in alphabetical order ==========*/

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="addfacetvertex">-</a>

  qh_addfacetvertex( qh, facet, newvertex )
    add newvertex to facet.vertices if not already there
    vertices are inverse sorted by vertex->id

  returns:
    True if new vertex for facet

  notes:
    see qh_replacefacetvertex
*/
boolT qh_addfacetvertex(qhT *qh, facetT *facet, vertexT *newvertex) {
  vertexT *vertex;
  int vertex_i, vertex_n;
  boolT isnew= True;

  FOREACHvertex_i_(qh, facet->vertices) {
    if (vertex->id < newvertex->id) {
      break;
    }else if (vertex->id == newvertex->id) {
      isnew= False;
      break;
    }
  }
  if (isnew)
    qh_setaddnth(qh, &facet->vertices, vertex_i, newvertex);
  return isnew;
} /* addfacetvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="addhash">-</a>

  qh_addhash( newelem, hashtable, hashsize, hash )
    add newelem to linear hash table at hash if not already there
*/
void qh_addhash(void *newelem, setT *hashtable, int hashsize, int hash) {
  int scan;
  void *elem;

  for (scan= (int)hash; (elem= SETelem_(hashtable, scan));
       scan= (++scan >= hashsize ? 0 : scan)) {
    if (elem == newelem)
      break;
  }
  /* loop terminates because qh_HASHfactor >= 1.1 by qh_initbuffers */
  if (!elem)
    SETelem_(hashtable, scan)= newelem;
} /* addhash */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_bestdist">-</a>

  qh_check_bestdist(qh)
    check that all points are within max_outside of the nearest facet
    if qh.ONLYgood,
      ignores !good facets

  see:
    qh_check_maxout(), qh_outerinner()

  notes:
    only called from qh_check_points()
      seldom used since qh.MERGING is almost always set
    if notverified>0 at end of routine
      some points were well inside the hull.  If the hull contains
      a lens-shaped component, these points were not verified.  Use
      options 'Qi Tv' to verify all points.  (Exhaustive check also verifies)

  design:
    determine facet for each point (if any)
    for each point
      start with the assigned facet or with the first facet
      find the best facet for the point and check all coplanar facets
      error if point is outside of facet
*/
void qh_check_bestdist(qhT *qh) {
  boolT waserror= False, unassigned;
  facetT *facet, *bestfacet, *errfacet1= NULL, *errfacet2= NULL;
  facetT *facetlist;
  realT dist, maxoutside, maxdist= -REALmax;
  pointT *point;
  int numpart= 0, facet_i, facet_n, notgood= 0, notverified= 0;
  setT *facets;

  trace1((qh, qh->ferr, 1020, "qh_check_bestdist: check points below nearest facet.  Facet_list f%d\n",
      qh->facet_list->id));
  maxoutside= qh_maxouter(qh);
  maxoutside += qh->DISTround;
  /* one more qh.DISTround for check computation */
  trace1((qh, qh->ferr, 1021, "qh_check_bestdist: check that all points are within %2.2g of best facet\n", maxoutside));
  facets= qh_pointfacet(qh /*qh.facet_list*/);
  if (!qh_QUICKhelp && qh->PRINTprecision)
    qh_fprintf(qh, qh->ferr, 8091, "\n\
qhull output completed.  Verifying that %d points are\n\
below %2.2g of the nearest %sfacet.\n",
             qh_setsize(qh, facets), maxoutside, (qh->ONLYgood ?  "good " : ""));
  FOREACHfacet_i_(qh, facets) {  /* for each point with facet assignment */
    if (facet)
      unassigned= False;
    else {
      unassigned= True;
      facet= qh->facet_list;
    }
    point= qh_point(qh, facet_i);
    if (point == qh->GOODpointp)
      continue;
    qh_distplane(qh, point, facet, &dist);
    numpart++;
    bestfacet= qh_findbesthorizon(qh, !qh_IScheckmax, point, facet, qh_NOupper, &dist, &numpart);
    /* occurs after statistics reported */
    maximize_(maxdist, dist);
    if (dist > maxoutside) {
      if (qh->ONLYgood && !bestfacet->good
      && !((bestfacet= qh_findgooddist(qh, point, bestfacet, &dist, &facetlist))
      && dist > maxoutside))
        notgood++;
      else {
        waserror= True;
        qh_fprintf(qh, qh->ferr, 6109, "qhull precision error: point p%d is outside facet f%d, distance= %6.8g maxoutside= %6.8g\n",
                facet_i, bestfacet->id, dist, maxoutside);
        if (errfacet1 != bestfacet) {
          errfacet2= errfacet1;
          errfacet1= bestfacet;
        }
      }
    }else if (unassigned && dist < -qh->MAXcoplanar)
      notverified++;
  }
  qh_settempfree(qh, &facets);
  if (notverified && !qh->DELAUNAY && !qh_QUICKhelp && qh->PRINTprecision)
    qh_fprintf(qh, qh->ferr, 8092, "\n%d points were well inside the hull.  If the hull contains\n\
a lens-shaped component, these points were not verified.  Use\n\
options 'Qci Tv' to verify all points.\n", notverified);
  if (maxdist > qh->outside_err) {
    qh_fprintf(qh, qh->ferr, 6110, "qhull precision error (qh_check_bestdist): a coplanar point is %6.2g from convex hull.  The maximum value(qh.outside_err) is %6.2g\n",
              maxdist, qh->outside_err);
    qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2);
  }else if (waserror && qh->outside_err > REALmax/2)
    qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2);
  /* else if waserror, the error was logged to qh.ferr but does not effect the output */
  trace0((qh, qh->ferr, 20, "qh_check_bestdist: max distance outside %2.2g\n", maxdist));
} /* check_bestdist */

#ifndef qh_NOmerge
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_maxout">-</a>

  qh_check_maxout(qh)
    updates qh.max_outside by checking all points against bestfacet
    if qh.ONLYgood, ignores !good facets

  returns:
    updates facet->maxoutside via qh_findbesthorizon()
    sets qh.maxoutdone
    if printing qh.min_vertex (qh_outerinner),
      it is updated to the current vertices
    removes inside/coplanar points from coplanarset as needed

  notes:
    defines coplanar as min_vertex instead of MAXcoplanar
    may not need to check near-inside points because of qh.MAXcoplanar
      and qh.KEEPnearinside (before it was -DISTround)

  see also:
    qh_check_bestdist()

  design:
    if qh.min_vertex is needed
      for all neighbors of all vertices
        test distance from vertex to neighbor
    determine facet for each point (if any)
    for each point with an assigned facet
      find the best facet for the point and check all coplanar facets
        (updates outer planes)
    remove near-inside points from coplanar sets
*/
void qh_check_maxout(qhT *qh) {
  facetT *facet, *bestfacet, *neighbor, **neighborp, *facetlist, *maxbestfacet= NULL, *minfacet, *maxfacet, *maxpointfacet;
  realT dist, maxoutside, mindist, nearest, old_maxoutside;
  realT maxoutside_base, minvertex_base;
  pointT *point, *maxpoint= NULL;
  int numpart= 0, facet_i, facet_n, notgood= 0;
  setT *facets, *vertices;
  vertexT *vertex, *minvertex;

  trace1((qh, qh->ferr, 1022, "qh_check_maxout: check and update qh.min_vertex %2.2g and qh.max_outside %2.2g\n", qh->min_vertex, qh->max_outside));
  minvertex_base= fmin_(qh->min_vertex, -(qh->ONEmerge+qh->DISTround));
  maxoutside= mindist= 0;
  minvertex= qh->vertex_list;
  maxfacet= minfacet= maxpointfacet= qh->facet_list;
  if (qh->VERTEXneighbors
  && (qh->PRINTsummary || qh->KEEPinside || qh->KEEPcoplanar
        || qh->TRACElevel || qh->PRINTstatistics || qh->VERIFYoutput || qh->CHECKfrequently
        || qh->PRINTout[0] == qh_PRINTsummary || qh->PRINTout[0] == qh_PRINTnone)) {
    trace1((qh, qh->ferr, 1023, "qh_check_maxout: determine actual minvertex\n"));
    vertices= qh_pointvertex(qh /*qh.facet_list*/);
    FORALLvertices {
      FOREACHneighbor_(vertex) {
        zinc_(Zdistvertex);  /* distance also computed by main loop below */
        qh_distplane(qh, vertex->point, neighbor, &dist);
        if (dist < mindist) {
          if (qh->min_vertex/minvertex_base > qh_WIDEmaxoutside && (qh->PRINTprecision || !qh->ALLOWwidemaxout)) {
            nearest= qh_vertex_bestdist(qh, neighbor->vertices);
            qh_fprintf(qh, qh->ferr, 7083, "Qhull precision warning: in post-processing (qh_check_maxout) p%d(v%d) is %2.2g below f%d nearest vertices %2.2g.  May be due to nearly adjacent vertices in 4-d and higher\n",
              qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id, nearest);
          }
          mindist= dist;
          minvertex= vertex;
          minfacet= neighbor;
        }
        if (-dist > qh->TRACEdist || dist > qh->TRACEdist
        || neighbor == qh->tracefacet || vertex == qh->tracevertex) {
          nearest= qh_vertex_bestdist(qh, neighbor->vertices);
          qh_fprintf(qh, qh->ferr, 8093, "qh_check_maxout: p%d(v%d) is %.2g from f%d nearest vertices %2.2g\n",
                    qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id, nearest);
        }
      }
    }
    if (qh->MERGING) {
      wmin_(Wminvertex, qh->min_vertex);
    }
    qh->min_vertex= mindist;
    qh_settempfree(qh, &vertices);
  }
  trace1((qh, qh->ferr, 1055, "qh_check_maxout: determine actual maxoutside\n"));
  maxoutside_base= fmax_(qh->max_outside, qh->ONEmerge+qh->DISTround);
  facets= qh_pointfacet(qh /*qh.facet_list*/);
  do {
    old_maxoutside= fmax_(qh->max_outside, maxoutside);
    FOREACHfacet_i_(qh, facets) {     /* for each point with facet assignment */
      if (facet) {
        point= qh_point(qh, facet_i);
        if (point == qh->GOODpointp)
          continue;
        zzinc_(Ztotcheck);
        qh_distplane(qh, point, facet, &dist);
        numpart++;
        bestfacet= qh_findbesthorizon(qh, qh_IScheckmax, point, facet, !qh_NOupper, &dist, &numpart);
        if (bestfacet && dist >= maxoutside) { /* FIXUP review '>=' vs. '>' for maxoutside */
          if (qh->ONLYgood && !bestfacet->good
          && !((bestfacet= qh_findgooddist(qh, point, bestfacet, &dist, &facetlist))
          && dist > maxoutside)) {       /* FIXUP '>=' ? */
            notgood++;
          }else if (dist/maxoutside_base > qh_WIDEmaxoutside && (qh->PRINTprecision || !qh->ALLOWwidemaxout)) {
            nearest= qh_vertex_bestdist(qh, bestfacet->vertices);
            if (nearest < fmax_(qh->ONEmerge, qh->max_outside) * qh_RATIOcoplanaroutside * 2) {
              qh_fprintf(qh, qh->ferr, 32, "Qhull precision warning: in post-processing (qh_check_maxout) p%d for f%d is %2.2g above twisted facet f%d nearest vertices %2.2g\n",
                qh_pointid(qh, point), facet->id, dist, bestfacet->id, nearest);
            }else {
              qh_fprintf(qh, qh->ferr, 33, "Qhull precision warning: in post-processing (qh_check_maxout) p%d for f%d is %2.2g above hidden facet f%d nearest vertices %2.2g\n",
                qh_pointid(qh, point), facet->id, dist, bestfacet->id, nearest);
            }
            maxbestfacet= bestfacet;
          }
          maxoutside= dist;
          maxfacet= bestfacet;
          maxpoint= point;
          maxpointfacet= facet;
        }
        if (dist > qh->TRACEdist || (bestfacet && bestfacet == qh->tracefacet))
          qh_fprintf(qh, qh->ferr, 8094, "qh_check_maxout: p%d is %.2g above f%d\n",
          qh_pointid(qh, point), dist, (bestfacet ? bestfacet->id : UINT_MAX));
      }
    }
  }while
    (maxoutside > 2*old_maxoutside);
    /* FIXUP no longer works why?  if qh.max_outside increases substantially, qh_SEARCHdist is not valid
          e.g., RBOX 5000 s Z1 G1e-13 t1001200614 | qhull */
  zzadd_(Zcheckpart, numpart);
  qh_settempfree(qh, &facets);
  wval_(Wmaxout)= maxoutside - qh->max_outside;
  wmax_(Wmaxoutside, qh->max_outside);
  if (!qh->APPROXhull && maxoutside > qh->DISTround) { /* initial value for f.maxoutside */
    FORALLfacets {
      if (maxoutside < facet->maxoutside) {
        if (!qh->KEEPcoplanar) {
          maxoutside= facet->maxoutside;
        }else {
          qh_fprintf(qh, qh->ferr, 7082, "Qhull precision warning (qh_check_maxout): f%d.maxoutside (%4.4g) is greater than computed qh.max_outside (%2.2g).  It should be less than or equal\n",
            facet->id, facet->maxoutside, maxoutside);  /* FIXUP -- how to report, what if lots? */
        }
      }
    }
  }
  qh->max_outside= maxoutside;
  qh_nearcoplanar(qh /*qh.facet_list*/);
  qh->maxoutdone= True;
  trace1((qh, qh->ferr, 1024, "qh_check_maxout:  p%d(v%d) is qh.min_vertex %2.2g below facet f%d.  Point p%d for f%d is qh.max_outside %2.2g above f%d.  %d points are outside of not-good facets\n", 
    qh_pointid(qh, minvertex->point), minvertex->id, qh->min_vertex, minfacet->id, qh_pointid(qh, maxpoint), maxpointfacet->id, qh->max_outside, maxfacet->id, notgood));
  if(!qh->ALLOWwidemaxout) {
    if (maxoutside/maxoutside_base > qh_WIDEmaxoutside) {
      qh_fprintf(qh, qh->ferr, 6297, "Qhull precision error (qh_check_maxout): large increase in qh.max_outside during post-processing dist %2.2g (%.1fx).  See warning QH0032/QH0033.  Disable with 'Q15' (allow_widemax) and 'Pp'\n",
        maxoutside, maxoutside/maxoutside_base);
      qh_printstats(qh, qh->ferr, qh->qhstat.precision, NULL);
      qh_errexit(qh, qh_ERRprec, maxbestfacet, NULL);
    }else if (!qh->APPROXhull && maxoutside_base > (qh->ONEmerge * qh_WIDEmaxoutside2)) {
      if (maxoutside > (qh->ONEmerge * qh_WIDEmaxoutside2)) {  /* wide facets may have been deleted */
        qh_fprintf(qh, qh->ferr, 6298, "Qhull precision error (qh_check_maxout): a facet merge, vertex merge, vertex, or coplanar point produced a wide facet %2.2g (%.1fx). Trace with option 'TWn' to identify the merge.   Allow with 'Q15' (allow_widemax)\n",
          maxoutside_base, maxoutside_base/(qh->ONEmerge + qh->DISTround));
        qh_printstats(qh, qh->ferr, qh->qhstat.precision, NULL);  /* FIXUP how to print precision stats */
        qh_errexit(qh, qh_ERRprec, maxbestfacet, NULL);
      }
    }else if (qh->min_vertex/minvertex_base > qh_WIDEmaxoutside) {
      /* FIXUP use QH00? */
      qh_fprintf(qh, qh->ferr, 6305, "Qhull precision error (qh_check_maxout): large increase in qh.min_vertex during post-processing dist %2.2g (%.1fx).  See warning QH7083.  Allow with 'Q15' (allow_widemax) and 'Pp'\n",
        qh->min_vertex, qh->min_vertex/minvertex_base);
      qh_printstats(qh, qh->ferr, qh->qhstat.precision, NULL);
      qh_errexit(qh, qh_ERRprec, minfacet, NULL);
    }else if (minvertex_base < -(qh->ONEmerge * qh_WIDEmaxoutside2)) {
      if (qh->min_vertex < -(qh->ONEmerge * qh_WIDEmaxoutside2)) {  /* wide facets may have been deleted */
        qh_fprintf(qh, qh->ferr, 6306, "Qhull precision error (qh_check_maxout): a facet or vertex merge produced a wide facet: v%d below f%d distance %2.2g (%.1fx). Trace with option 'TWn' to identify the merge.  Allow with 'Q15' (allow_widemax)\n",
          minvertex->id, minfacet->id, mindist, -qh->min_vertex/(qh->ONEmerge + qh->DISTround));
        qh_printstats(qh, qh->ferr, qh->qhstat.precision, NULL);  /* FIXUP how to print precision stats */
        qh_errexit(qh, qh_ERRprec, minfacet, NULL);
      }
    }
  }
} /* check_maxout */
#else /* qh_NOmerge */
void qh_check_maxout(qhT *qh) {
}
#endif

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_output">-</a>

  qh_check_output(qh)
    performs the checks at the end of qhull algorithm
    Maybe called after voronoi output.  Will recompute otherwise centrums are Voronoi centers instead
*/
void qh_check_output(qhT *qh) {
  int i;

  if (qh->STOPcone)
    return;
  if (qh->VERIFYoutput | qh->IStracing | qh->CHECKfrequently) {
    qh_checkpolygon(qh, qh->facet_list);
    qh_checkflipped_all(qh, qh->facet_list);
    qh_checkconvex(qh, qh->facet_list, qh_ALGORITHMfault);
  }else if (!qh->MERGING && qh_newstats(qh, qh->qhstat.precision, &i)) {
    qh_checkflipped_all(qh, qh->facet_list);
    qh_checkconvex(qh, qh->facet_list, qh_ALGORITHMfault);
  }
} /* check_output */



/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_point">-</a>

  qh_check_point(qh, point, facet, maxoutside, maxdist, errfacet1, errfacet2 )
    check that point is less than maxoutside from facet
*/
void qh_check_point(qhT *qh, pointT *point, facetT *facet, realT *maxoutside, realT *maxdist, facetT **errfacet1, facetT **errfacet2) {
  realT dist, nearest;

  /* occurs after statistics reported */
  qh_distplane(qh, point, facet, &dist);
  if (dist > *maxoutside) {
    if (*errfacet1 != facet) {
      *errfacet2= *errfacet1;
      *errfacet1= facet;
    }
    nearest= qh_vertex_bestdist(qh, facet->vertices);
    qh_fprintf(qh, qh->ferr, 6111, "qhull precision error: point p%d is outside facet f%d, distance= %6.8g maxoutside= %6.8g nearest vertices %2.2g\n",
              qh_pointid(qh, point), facet->id, dist, *maxoutside, nearest);
  }
  maximize_(*maxdist, dist);
} /* qh_check_point */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_points">-</a>

  qh_check_points(qh)
    checks that all points are inside all facets

  notes:
    if many points and qh_check_maxout not called (i.e., !qh.MERGING),
       calls qh_findbesthorizon (seldom done).
    ignores flipped facets
    maxoutside includes 2 qh.DISTrounds
      one qh.DISTround for the computed distances in qh_check_points
    qh_printafacet and qh_printsummary needs only one qh.DISTround
    the computation for qh.VERIFYdirect does not account for qh.other_points

  design:
    if many points
      use qh_check_bestdist()
    else
      for all facets
        for all points
          check that point is inside facet
*/
void qh_check_points(qhT *qh) {
  facetT *facet, *errfacet1= NULL, *errfacet2= NULL;
  realT total, maxoutside, maxdist= -REALmax;
  pointT *point, **pointp, *pointtemp;
  boolT testouter;

  maxoutside= qh_maxouter(qh);
  maxoutside += qh->DISTround;
  /* one more qh.DISTround for check computation */
  trace1((qh, qh->ferr, 1025, "qh_check_points: check all points below %2.2g of all facet planes\n",
          maxoutside));
  if (qh->num_good)   /* miss counts other_points and !good facets */
     total= (float)qh->num_good * (float)qh->num_points;
  else
     total= (float)qh->num_facets * (float)qh->num_points;
  if (total >= qh_VERIFYdirect && !qh->maxoutdone) {
    if (!qh_QUICKhelp && qh->SKIPcheckmax && qh->MERGING)
      qh_fprintf(qh, qh->ferr, 7075, "qhull input warning: merging without checking outer planes('Q5' or 'Po').\n\
Verify may report that a point is outside of a facet.\n");
    qh_check_bestdist(qh);
  }else {
    if (qh_MAXoutside && qh->maxoutdone)
      testouter= True;
    else
      testouter= False;
    if (!qh_QUICKhelp) {
      if (qh->MERGEexact)
        qh_fprintf(qh, qh->ferr, 7076, "qhull input warning: exact merge ('Qx').  Verify may report that a point\n\
is outside of a facet.  See qh-optq.htm#Qx\n");
      else if (qh->SKIPcheckmax || qh->NOnearinside)
        qh_fprintf(qh, qh->ferr, 7077, "qhull input warning: no outer plane check ('Q5') or no processing of\n\
near-inside points ('Q8').  Verify may report that a point is outside\n\
of a facet.\n");
    }
    if (qh->PRINTprecision) {
      if (testouter)
        qh_fprintf(qh, qh->ferr, 8098, "\n\
Output completed.  Verifying that all points are below outer planes of\n\
all %sfacets.  Will make %2.0f distance computations.\n",
              (qh->ONLYgood ?  "good " : ""), total);
      else
        qh_fprintf(qh, qh->ferr, 8099, "\n\
Output completed.  Verifying that all points are below %2.2g of\n\
all %sfacets.  Will make %2.0f distance computations.\n",
              maxoutside, (qh->ONLYgood ?  "good " : ""), total);
    }
    FORALLfacets {
      if (!facet->good && qh->ONLYgood)
        continue;
      if (facet->flipped)
        continue;
      if (!facet->normal) {
        qh_fprintf(qh, qh->ferr, 7061, "qhull warning (qh_check_points): missing normal for facet f%d\n", facet->id);
        continue;
      }
      if (testouter) {
#if qh_MAXoutside
        maxoutside= facet->maxoutside + 2* qh->DISTround;
        /* one DISTround to actual point and another to computed point */
#endif
      }
      FORALLpoints {
        if (point != qh->GOODpointp)
          qh_check_point(qh, point, facet, &maxoutside, &maxdist, &errfacet1, &errfacet2);
      }
      FOREACHpoint_(qh->other_points) {
        if (point != qh->GOODpointp)
          qh_check_point(qh, point, facet, &maxoutside, &maxdist, &errfacet1, &errfacet2);
      }
    }
    if (maxdist > qh->outside_err) {
      qh_fprintf(qh, qh->ferr, 6112, "qhull precision error (qh_check_points): a coplanar point is %6.2g from convex hull.  The maximum value(qh.outside_err) is %6.2g\n",
                maxdist, qh->outside_err );
      qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2 );
    }else if (errfacet1 && qh->outside_err > REALmax/2)
        qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2 );
    /* else if errfacet1, the error was logged to qh.ferr but does not effect the output */
    trace0((qh, qh->ferr, 21, "qh_check_points: max distance outside %2.2g\n", maxdist));
  }
} /* check_points */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkconvex">-</a>

  qh_checkconvex(qh, facetlist, fault )
    check that each ridge in facetlist is convex
    fault = qh_DATAfault if reporting errors
          = qh_ALGORITHMfault otherwise

  returns:
    counts Zconcaveridges and Zcoplanarridges
    errors if concaveridge or if merging an coplanar ridge

  notes:
    if not merging,
      tests vertices for neighboring simplicial facets
    else if ZEROcentrum,
      tests vertices for neighboring simplicial   facets
    else
      tests centrums of neighboring facets

  design:
    for all facets
      report flipped facets
      if ZEROcentrum and simplicial neighbors
        test vertices for neighboring simplicial facets
      else
        test centrum against all neighbors
*/
void qh_checkconvex(qhT *qh, facetT *facetlist, int fault) {
  facetT *facet, *neighbor, **neighborp, *errfacet1=NULL, *errfacet2=NULL;
  vertexT *vertex;
  realT dist;
  pointT *centrum;
  boolT waserror= False, centrum_warning= False, tempcentrum= False, allsimplicial;
  int neighbor_i;

  trace1((qh, qh->ferr, 1026, "qh_checkconvex: check all ridges are convex\n"));
  if (!qh->RERUN) {
    zzval_(Zconcaveridges)= 0;
    zzval_(Zcoplanarridges)= 0;
  }
  FORALLfacet_(facetlist) {
    if (facet->flipped) {
      qh_joggle_restart(qh, "flipped facet");
      qh_fprintf(qh, qh->ferr, 6113, "qhull precision error: f%d is flipped(interior point is outside)\n",
               facet->id);
      errfacet1= facet;
      waserror= True;
      continue;
    }
    if (qh->MERGING && (!qh->ZEROcentrum || !facet->simplicial || facet->tricoplanar))
      allsimplicial= False;
    else {
      allsimplicial= True;
      neighbor_i= 0;
      FOREACHneighbor_(facet) {
        vertex= SETelemt_(facet->vertices, neighbor_i++, vertexT);
        if (!neighbor->simplicial || neighbor->tricoplanar) {
          allsimplicial= False;
          continue;
        }
        qh_distplane(qh, vertex->point, neighbor, &dist);
        if (dist > -qh->DISTround) {
          if (fault == qh_DATAfault) {
            qh_joggle_restart(qh, "coplanar or concave ridge");
            qh_fprintf(qh, qh->ferr, 6114, "qhull precision error: initial simplex is not convex. Distance=%.2g\n", dist);
            qh_errexit(qh, qh_ERRsingular, NULL, NULL);
          }
          if (dist > qh->DISTround) {
            zzinc_(Zconcaveridges);
            qh_joggle_restart(qh, "concave ridge");
            qh_fprintf(qh, qh->ferr, 6115, "qhull precision error: f%d is concave to f%d, since p%d(v%d) is %6.4g above\n",
              facet->id, neighbor->id, qh_pointid(qh, vertex->point), vertex->id, dist);
            errfacet1= facet;
            errfacet2= neighbor;
            waserror= True;
          }else if (qh->ZEROcentrum) {
            if (dist > 0) {     /* qh_checkzero checks that dist < - qh->DISTround */
              zzinc_(Zcoplanarridges);
              qh_joggle_restart(qh, "coplanar ridge");
              qh_fprintf(qh, qh->ferr, 6116, "qhull precision error: f%d is clearly not convex to f%d, since p%d(v%d) is %6.4g above\n",
                facet->id, neighbor->id, qh_pointid(qh, vertex->point), vertex->id, dist);
              errfacet1= facet;
              errfacet2= neighbor;
              waserror= True;
            }
          }else {
            zzinc_(Zcoplanarridges);
            qh_joggle_restart(qh, "coplanar ridge");
            trace0((qh, qh->ferr, 22, "qhull precision error: f%d may be coplanar to f%d, since p%d(v%d) is within %6.4g during p%d\n",
              facet->id, neighbor->id, qh_pointid(qh, vertex->point), vertex->id, dist, qh->furthest_id));
          }
        }
      }
    }
    if (!allsimplicial && qh->hull_dim <= 3) {  /* FIXUP centrum verification for non-simplicial merge */
      if (qh->CENTERtype == qh_AScentrum) {
        if (!facet->center)
          facet->center= qh_getcentrum(qh, facet);
        centrum= facet->center;
      }else {
        if (!centrum_warning && (!facet->simplicial || facet->tricoplanar)) {
           centrum_warning= True;
           qh_fprintf(qh, qh->ferr, 7062, "qhull warning: recomputing centrums for convexity test.  This may lead to false, precision errors.\n");
        }
        centrum= qh_getcentrum(qh, facet);
        tempcentrum= True;
      }
      FOREACHneighbor_(facet) {
        if (qh->ZEROcentrum && facet->simplicial && neighbor->simplicial)
          continue;
        if (facet->tricoplanar || neighbor->tricoplanar)
          continue;
        zzinc_(Zdistconvex);
        qh_distplane(qh, centrum, neighbor, &dist);
        if (dist > qh->DISTround) {
          zzinc_(Zconcaveridges);
          qh_joggle_restart(qh, "concave ridge");
          qh_fprintf(qh, qh->ferr, 6117, "qhull precision error: f%d is concave to f%d.  Centrum of f%d is %6.4g above f%d\n",
            facet->id, neighbor->id, facet->id, dist, neighbor->id);
          errfacet1= facet;
          errfacet2= neighbor;
          waserror= True;
        }else if (dist >= 0.0) {   /* if arithmetic always rounds the same,
                                     can test against centrum radius instead */
          zzinc_(Zcoplanarridges);
          qh_joggle_restart(qh, "coplanar ridge");
          qh_fprintf(qh, qh->ferr, 6118, "qhull precision error: f%d is coplanar or concave to f%d.  Centrum of f%d is %6.4g above f%d\n",
            facet->id, neighbor->id, facet->id, dist, neighbor->id);
          errfacet1= facet;
          errfacet2= neighbor;
          waserror= True;
        }
      }
      if (tempcentrum)
        qh_memfree(qh, centrum, qh->normal_size);
    }
  }
  if (waserror && !qh->FORCEoutput)
    qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2);
} /* checkconvex */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkfacet">-</a>

  qh_checkfacet(qh, facet, newmerge, waserror )
    checks for consistency errors in facet
    newmerge set if from merge_r.c

  returns:
    sets waserror if any error occurs

  checks:
    vertex ids are inverse sorted
    unless newmerge, at least hull_dim neighbors and vertices (exactly if simplicial)
    if non-simplicial, at least as many ridges as neighbors
    neighbors are not duplicated
    ridges are not duplicated
    in 3-d, ridges=verticies
    (qh.hull_dim-1) ridge vertices
    neighbors are reciprocated
    ridge neighbors are facet neighbors and a ridge for every neighbor
    simplicial neighbors match facetintersect
    vertex intersection matches vertices of common ridges
    vertex neighbors and facet vertices agree
    all ridges have distinct vertex sets

  notes:
    uses neighbor->seen

  design:
    check sets
    check vertices
    check sizes of neighbors and vertices
    check for qh_MERGEridge and qh_DUPLICATEridge flags
    check neighbor set
    check ridge set
    check ridges, neighbors, and vertices
*/
void qh_checkfacet(qhT *qh, facetT *facet, boolT newmerge, boolT *waserrorp) {
  facetT *neighbor, **neighborp, *errother=NULL;
  ridgeT *ridge, **ridgep, *errridge= NULL, *ridge2;
  vertexT *vertex, **vertexp;
  unsigned previousid= INT_MAX;
  int numneighbors, numvertices, numridges=0, numRvertices=0;
  boolT waserror= False;
  int skipA, skipB, ridge_i, ridge_n, i, last_v= qh->hull_dim-2;
  setT *intersection;

  trace4((qh, qh->ferr, 4088, "qh_checkfacet: check f%d newmerge? %d\n", facet->id, newmerge));
  if (facet->visible && !qh->NEWtentative) {
    qh_fprintf(qh, qh->ferr, 6119, "qhull internal error (qh_checkfacet): facet f%d is on qh.visible_list\n",
      facet->id);
    qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }
  if (facet->redundant && !facet->visible && qh_setsize(qh, qh->degen_mergeset)==0) {
    qh_fprintf(qh, qh->ferr, 6327, "qhull internal error (qh_checkfacet): redundant facet f%d not on qh.visible_list\n",
      facet->id);
    waserror= True;
  }
  if (facet->degenerate && !facet->visible && qh_setsize(qh, qh->degen_mergeset)==0) { 
    qh_fprintf(qh, qh->ferr, 6328, "qhull internal error (qh_checkfacet): degenerate facet f%d is not on qh.visible_list and qh.degen_mergeset is empty\n",
      facet->id);
    waserror= True;
  }
  if (!facet->normal) {
    qh_fprintf(qh, qh->ferr, 6120, "qhull internal error (qh_checkfacet): facet f%d does not have  a normal\n",
      facet->id);
    waserror= True;
  }
  if (!facet->newfacet) {
    if (facet->dupridge) {
      qh_fprintf(qh, qh->ferr, 6304, "qhull internal error (qh_checkfacet): f%d is 'dupridge' but it is not a newfacet\n",  facet->id);
      waserror= True;
    }
    if (facet->newmerge) {
      qh_fprintf(qh, qh->ferr, 6311, "qhull internal error (qh_checkfacet): f%d is 'newmerge' but it is not a newfacet.  Missing call to qh_reducevertices\n",  facet->id, getid_(qh->newfacet_list));
      waserror= True;
    }
  }
  qh_setcheck(qh, facet->vertices, "vertices for f", facet->id);
  qh_setcheck(qh, facet->ridges, "ridges for f", facet->id);
  qh_setcheck(qh, facet->outsideset, "outsideset for f", facet->id);
  qh_setcheck(qh, facet->coplanarset, "coplanarset for f", facet->id);
  qh_setcheck(qh, facet->neighbors, "neighbors for f", facet->id);
  FOREACHvertex_(facet->vertices) {
    if (vertex->deleted) {
      qh_fprintf(qh, qh->ferr, 6121, "qhull internal error (qh_checkfacet): deleted vertex v%d in f%d\n", vertex->id, facet->id);
      qh_errprint(qh, "ERRONEOUS", NULL, NULL, NULL, vertex);
      waserror= True;
    }
    if (vertex->id >= previousid) {
      qh_fprintf(qh, qh->ferr, 6122, "qhull internal error (qh_checkfacet): vertices of f%d are not in descending id order at v%d\n", facet->id, vertex->id);
      waserror= True;
      break;
    }
    previousid= vertex->id;
  }
  numneighbors= qh_setsize(qh, facet->neighbors);
  numvertices= qh_setsize(qh, facet->vertices);
  numridges= qh_setsize(qh, facet->ridges);
  if (facet->simplicial) {
    if (numvertices+numneighbors != 2*qh->hull_dim
    && !facet->degenerate && !facet->redundant) {
      qh_fprintf(qh, qh->ferr, 6123, "qhull internal error (qh_checkfacet): for simplicial facet f%d, #vertices %d + #neighbors %d != 2*qh->hull_dim\n",
                facet->id, numvertices, numneighbors);
      qh_setprint(qh, qh->ferr, "", facet->neighbors);
      waserror= True;
    }
  }else { /* non-simplicial */
    if (!newmerge
    &&(numvertices < qh->hull_dim || numneighbors < qh->hull_dim)
    && !facet->degenerate && !facet->redundant) {
      qh_fprintf(qh, qh->ferr, 6124, "qhull internal error (qh_checkfacet): for facet f%d, #vertices %d or #neighbors %d < qh->hull_dim\n",
         facet->id, numvertices, numneighbors);
       waserror= True;
    }
    /* in 3-d, can get a vertex twice in an edge list, e.g., RBOX 1000 s W1e-13 t995849315 D2 | QHULL d Tc Tv TP624 TW1e-13 T4 */
    if (numridges < numneighbors
    ||(qh->hull_dim == 3 && numvertices > numridges && !qh->NEWfacets)
    ||(qh->hull_dim == 2 && numridges + numvertices + numneighbors != 6)) {
      if (!facet->degenerate && !facet->redundant) {
        qh_fprintf(qh, qh->ferr, 6125, "qhull internal error (qh_checkfacet): for facet f%d, #ridges %d < #neighbors %d or(3-d) > #vertices %d or(2-d) not all 2\n",
            facet->id, numridges, numneighbors, numvertices);
        waserror= True;
      }
    }
  }
  FOREACHneighbor_(facet) {
    if (neighbor == qh_MERGEridge || neighbor == qh_DUPLICATEridge) {
      qh_fprintf(qh, qh->ferr, 6126, "qhull internal error (qh_checkfacet): facet f%d still has a MERGEridge or DUPLICATEridge neighbor\n", facet->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    if (neighbor->visible) {
      qh_fprintf(qh, qh->ferr, 6329, "qhull internal error (qh_checkfacet): facet f%d has deleted neighbor f%d (qh.visible_list)\n",
        facet->id, neighbor->id);
      errother= neighbor;
      waserror= True;
    }
    neighbor->seen= True;
  }
  FOREACHneighbor_(facet) {
    if (!qh_setin(neighbor->neighbors, facet)) {
      qh_fprintf(qh, qh->ferr, 6127, "qhull internal error (qh_checkfacet): facet f%d has neighbor f%d, but f%d does not have neighbor f%d\n",
              facet->id, neighbor->id, neighbor->id, facet->id);
      errother= neighbor;
      waserror= True;
    }
    if (!neighbor->seen) {
      qh_fprintf(qh, qh->ferr, 6128, "qhull internal error (qh_checkfacet): facet f%d has a duplicate neighbor f%d\n",
              facet->id, neighbor->id);
      errother= neighbor;
      waserror= True;
    }
    neighbor->seen= False;
  }
  FOREACHridge_(facet->ridges) {
    qh_setcheck(qh, ridge->vertices, "vertices for r", ridge->id);
    ridge->seen= False;
  }
  FOREACHridge_(facet->ridges) {
    if (ridge->seen) {
      qh_fprintf(qh, qh->ferr, 6129, "qhull internal error (qh_checkfacet): facet f%d has a duplicate ridge r%d\n",
              facet->id, ridge->id);
      errridge= ridge;
      waserror= True;
    }
    ridge->seen= True;
    numRvertices= qh_setsize(qh, ridge->vertices);
    if (numRvertices != qh->hull_dim - 1) {
      qh_fprintf(qh, qh->ferr, 6130, "qhull internal error (qh_checkfacet): ridge between f%d and f%d has %d vertices\n",
                ridge->top->id, ridge->bottom->id, numRvertices);
      errridge= ridge;
      waserror= True;
    }
    neighbor= otherfacet_(ridge, facet);
    neighbor->seen= True;
    if (!qh_setin(facet->neighbors, neighbor)) {
      qh_fprintf(qh, qh->ferr, 6131, "qhull internal error (qh_checkfacet): for facet f%d, neighbor f%d of ridge r%d not in facet\n",
           facet->id, neighbor->id, ridge->id);
      errridge= ridge;
      waserror= True;
    }
    if (!facet->newfacet && !neighbor->newfacet) {
      if (!ridge->tested | ridge->nonconvex | ridge->mergevertex) {
        qh_fprintf(qh, qh->ferr, 6312, "qhull internal error (qh_checkfacet): ridge r%d is nonconvex (%d) mergevertex (%d) or not tested (%d) for facet f%d, neighbor f%d\n",
          ridge->id, ridge->nonconvex, ridge->mergevertex, ridge->tested, facet->id, neighbor->id);
        errridge= ridge;
        waserror= True;
        /* FIXUP test all facet flags */
      }

    }
  }
  if (!facet->simplicial) {
    FOREACHneighbor_(facet) {
      if (!neighbor->seen) {
        qh_fprintf(qh, qh->ferr, 6132, "qhull internal error (qh_checkfacet): facet f%d does not have a ridge for neighbor f%d\n",
              facet->id, neighbor->id);
        errother= neighbor;
        waserror= True;
      }
      intersection= qh_vertexintersect_new(qh, facet->vertices, neighbor->vertices);
      qh_settemppush(qh, intersection);
      FOREACHvertex_(facet->vertices) {
        vertex->seen= False;
        vertex->seen2= False;
      }
      FOREACHvertex_(intersection)
        vertex->seen= True;
      FOREACHridge_(facet->ridges) {
        if (neighbor != otherfacet_(ridge, facet))
            continue;
        FOREACHvertex_(ridge->vertices) {
          if (!vertex->seen) {
            qh_fprintf(qh, qh->ferr, 6133, "qhull internal error (qh_checkfacet): vertex v%d in r%d not in f%d intersect f%d\n",
                  vertex->id, ridge->id, facet->id, neighbor->id);
            qh_errexit(qh, qh_ERRqhull, facet, ridge);
          }
          vertex->seen2= True;
        }
      }
      if (!newmerge) {
        FOREACHvertex_(intersection) {
          if (!vertex->seen2) {
            if (!qh->MERGING) {
              qh_fprintf(qh, qh->ferr, 6134, "qhull precision error (qh_checkfacet): vertex v%d in f%d intersect f%d but not in a ridge.  Last point was p%d\n",
                     vertex->id, facet->id, neighbor->id, qh->furthest_id);
              if (!qh->FORCEoutput) {
                qh_errprint(qh, "ERRONEOUS", facet, neighbor, NULL, vertex);
                qh_errexit(qh, qh_ERRqhull, NULL, NULL);
              }
            }else {
              trace4((qh, qh->ferr, 4073, "qhull precision error (qh_checkfacet): vertex v%d in f%d intersect f%d but not in a ridge.  OK due to merging\n",
                vertex->id, facet->id, neighbor->id));
            }
          }
        }
      }
      qh_settempfree(qh, &intersection);
    }
  }else { /* simplicial */
    FOREACHneighbor_(facet) {
      if (neighbor->simplicial && !facet->degenerate && !neighbor->degenerate) {
        skipA= SETindex_(facet->neighbors, neighbor);
        skipB= qh_setindex(neighbor->neighbors, facet);
        if (skipA<0 || skipB<0 || !qh_setequal_skip(facet->vertices, skipA, neighbor->vertices, skipB)) {
          qh_fprintf(qh, qh->ferr, 6135, "qhull internal error (qh_checkfacet): facet f%d skip %d and neighbor f%d skip %d do not match \n",
                   facet->id, skipA, neighbor->id, skipB);
          errother= neighbor;
          waserror= True;
        }
      }
    }
  }
  if (qh->hull_dim < 5 && (qh->IStracing > 2 || qh->CHECKfrequently)) {
    FOREACHridge_i_(qh, facet->ridges) {           /* expensive */
      if (!ridge->mergevertex) {
        for (i=ridge_i+1; i < ridge_n; i++) {
          ridge2= SETelemt_(facet->ridges, i, ridgeT);
          if (SETelem_(ridge->vertices, last_v) == SETelem_(ridge2->vertices, last_v)) { /* SETfirst is likely to be the same */
            if (SETfirst_(ridge->vertices) == SETfirst_(ridge2->vertices)) {
              if (qh_setequal(ridge->vertices, ridge2->vertices)) {
                qh_fprintf(qh, qh->ferr, 6294, "qhull internal error (qh_checkfacet): ridges r%d and r%d (f%d) have the same vertices\n", /* same as duplicate ridge */
                  ridge->id, ridge2->id, facet->id);
                errridge= ridge;
                waserror= True;
              }
            }
          }
        }
      }
    }
  }
  if (waserror) {
    qh_errprint(qh, "ERRONEOUS", facet, errother, errridge, NULL);
    *waserrorp= True;
  }
} /* checkfacet */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkflipped_all">-</a>

  qh_checkflipped_all(qh, facetlist )
    checks orientation of facets in list against interior point
*/
void qh_checkflipped_all(qhT *qh, facetT *facetlist) {
  facetT *facet;
  boolT waserror= False;
  realT dist;

  if (facetlist == qh->facet_list)
    zzval_(Zflippedfacets)= 0;
  FORALLfacet_(facetlist) {
    if (facet->normal && !qh_checkflipped(qh, facet, &dist, !qh_ALL)) {
      qh_fprintf(qh, qh->ferr, 6136, "qhull precision error: facet f%d is flipped, distance= %6.12g\n",
              facet->id, dist);
      if (!qh->FORCEoutput) {
        qh_errprint(qh, "ERRONEOUS", facet, NULL, NULL, NULL);
        waserror= True;
      }
    }
  }
  if (waserror) {
    qh_fprintf(qh, qh->ferr, 8101, "\n\
A flipped facet occurs when its distance to the interior point is\n\
greater than %2.2g, the maximum roundoff error.\n", -qh->DISTround);
    qh_errexit(qh, qh_ERRprec, NULL, NULL);
  }
} /* checkflipped_all */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkpolygon">-</a>

  qh_checkpolygon(qh, facetlist )
    checks the correctness of the structure

  notes:
    call with either qh.facet_list or qh.newfacet_list
    checks num_facets and num_vertices if qh.facet_list

  design:
    for each facet
      checks facet and outside set
    initializes vertexlist
    for each facet
      checks vertex set
    if checking all facets(qh.facetlist)
      check facet count
      if qh.VERTEXneighbors
        check vertex neighbors and count
      check vertex count
*/
void qh_checkpolygon(qhT *qh, facetT *facetlist) {
  facetT *facet, *neighbor, **neighborp;
  facetT *errorfacet= NULL, *errorfacet2= NULL;
  vertexT *vertex, **vertexp, *vertexlist;
  int numfacets= 0, numvertices= 0, numridges= 0;
  int totvneighbors= 0, totvertices= 0;
  boolT waserror= False, newseen= False, newvertexseen= False, nextseen= False, visibleseen= False;
  boolT checkfacet;

  trace1((qh, qh->ferr, 1027, "qh_checkpolygon: check all facets from f%d\n", facetlist->id));
  if (facetlist != qh->facet_list || qh->ONLYgood)
    nextseen= True;
  FORALLfacet_(facetlist) {
    if (facet == qh->visible_list) {
      if(newseen){
        qh_fprintf(qh, qh->ferr, 6285, "qhull internal error (qh_checkpolygon): qh.visible_list f%d is after qh.newfacet_list f%d.  It should be at, before, or NULL\n",
          facet->id, getid_(qh->newfacet_list));
        qh_printlists(qh);
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
      }
      visibleseen= True;
    }
    if (facet == qh->newfacet_list)
      newseen= True;
    if (facet->newfacet && !newseen && !visibleseen) {
        qh_fprintf(qh, qh->ferr, 6289, "qhull internal error (qh_checkpolygon): f%d is 'newfacet' but it is not on qh.newfacet_list f%d or visible_list f%d\n",  facet->id, getid_(qh->newfacet_list), getid_(qh->visible_list));
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    if (!facet->newfacet && newseen) {
        qh_fprintf(qh, qh->ferr, 6292, "qhull internal error (qh_checkpolygon): f%d is on qh.newfacet_list f%d but it is not 'newfacet'\n",  facet->id, getid_(qh->newfacet_list));
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    if (facet->visible != (visibleseen & !newseen)) {
      if(facet->visible)
        qh_fprintf(qh, qh->ferr, 6290, "qhull internal error (qh_checkpolygon): f%d is 'visible' but it is not on qh.visible_list f%d\n", facet->id, getid_(qh->visible_list));
      else
        qh_fprintf(qh, qh->ferr, 6291, "qhull internal error (qh_checkpolygon): f%d is on qh.visible_list f%d but it is not 'visible'\n", facet->id, qh->newfacet_list->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    if (qh->NEWtentative) {
      checkfacet= !facet->newfacet;
    }else {
      checkfacet= !facet->visible;
    }
    if(checkfacet) {
      if (!nextseen) {
        if (facet == qh->facet_next)  /* previous facets do not have outsideset */
          nextseen= True;
        else if (qh_setsize(qh, facet->outsideset)) {
          if (!qh->NARROWhull
#if !qh_COMPUTEfurthest
          || facet->furthestdist >= qh->MINoutside
#endif
                        ) {
            qh_fprintf(qh, qh->ferr, 6137, "qhull internal error (qh_checkpolygon): f%d has outside points before qh->facet_next f%d\n",
                     facet->id, getid_(qh->facet_next));
            qh_errexit(qh, qh_ERRqhull, facet, NULL);
          }
        }
      }
      numfacets++;
      qh_checkfacet(qh, facet, False, &waserror);
    }
  }
  if (!newseen && qh->newfacet_list && qh->newfacet_list->next && facetlist == qh->facet_list) {
    qh_fprintf(qh, qh->ferr, 6286, "qhull internal error (qh_checkpolygon): qh.newfacet_list f%d is not on qh.facet_list f%d\n", 
      qh->newfacet_list->id, facetlist->id);
    qh_printlists(qh);
    qh_errexit(qh, qh_ERRqhull, qh->visible_list, NULL);
  }
  if (facetlist == qh->facet_list) {
    if (!visibleseen && qh->visible_list && qh->visible_list->next) {
      qh_fprintf(qh, qh->ferr, 6138, "qhull internal error (qh_checkpolygon): qh.visible_list f%d is not on qh.facet_list f%d\n", 
        qh->visible_list->id, facetlist->id);
      qh_printlists(qh);
      qh_errexit(qh, qh_ERRqhull, qh->visible_list, NULL);
    }
    vertexlist= qh->vertex_list;
  }else if (facetlist == qh->newfacet_list) {
    vertexlist= qh->newvertex_list;
  }else {
    vertexlist= NULL;
  }
  FORALLvertex_(vertexlist) {
    if(vertex == qh->newvertex_list)
      newvertexseen= True;
    vertex->seen= False;
    vertex->visitid= 0;
    if(vertex->newfacet && !newvertexseen && !vertex->deleted) {
      qh_fprintf(qh, qh->ferr, 6288, "qhull internal error (qh_checkpolygon): v%d is 'newfacet' but it is not on new vertex list v%d\n", vertex->id, getid_(qh->newvertex_list));
      qh_errexit(qh, qh_ERRqhull, qh->visible_list, NULL);
    }
  }
  if(!newvertexseen && qh->newvertex_list && qh->newvertex_list->next) {
    qh_fprintf(qh, qh->ferr, 6287, "qhull internal error (qh_checkpolygon): new vertex list v%d is not on vertex list\n", qh->newvertex_list->id);
    qh_printlists(qh);
    qh_errexit(qh, qh_ERRqhull, qh->visible_list, NULL);
  }
  FORALLfacet_(facetlist) {
    if (facet->visible)
      continue;
    if (facet->simplicial)
      numridges += qh->hull_dim;
    else
      numridges += qh_setsize(qh, facet->ridges);
    FOREACHvertex_(facet->vertices) {
      vertex->visitid++;
      if (!vertex->seen) {
        vertex->seen= True;
        numvertices++;
        if (qh_pointid(qh, vertex->point) == qh_IDunknown) {
          qh_fprintf(qh, qh->ferr, 6139, "qhull internal error (qh_checkpolygon): unknown point %p for vertex v%d first_point %p\n",
                   vertex->point, vertex->id, qh->first_point);
          waserror= True;
        }
      }
    }
  }
  qh->vertex_visit += (unsigned int)numfacets;
  if (facetlist == qh->facet_list) {
    if (numfacets != qh->num_facets - qh->num_visible) {
      qh_fprintf(qh, qh->ferr, 6140, "qhull internal error (qh_checkpolygon): actual number of facets is %d, cumulative facet count is %d - %d visible facets\n",
              numfacets, qh->num_facets, qh->num_visible);
      waserror= True;
    }
    qh->vertex_visit++;
    if (qh->VERTEXneighbors) {
      FORALLvertices {
        qh_setcheck(qh, vertex->neighbors, "neighbors for v", vertex->id);
        if (vertex->deleted)
          continue;
        totvneighbors += qh_setsize(qh, vertex->neighbors);
      }
      FORALLfacet_(facetlist) {
        if (!facet->visible)
          totvertices += qh_setsize(qh, facet->vertices);
      }
      if (totvneighbors != totvertices) {
        qh_fprintf(qh, qh->ferr, 6141, "qhull internal error (qh_checkpolygon): vertex neighbors inconsistent.  Maybe duplicate or missing. Totvneighbors %d, totvertices %d\n",
                totvneighbors, totvertices);
        FORALLvertices {
          if (vertex->deleted)
            continue;
          qh->visit_id++;
          FOREACHneighbor_(vertex) {
            if (neighbor->visitid==qh->visit_id) {
              qh_fprintf(qh, qh->ferr, 6275, "qhull internal error (qh_checkpolygon): facet f%d occurs twice in neighbors of vertex v%d\n",
                  neighbor->id, vertex->id);
              errorfacet2= errorfacet;
              errorfacet= neighbor;
            }
            neighbor->visitid= qh->visit_id;
            if (!qh_setin(neighbor->vertices, vertex)) {
              qh_fprintf(qh, qh->ferr, 6276, "qhull internal error (qh_checkpolygon): facet f%d is a neighbor of vertex v%d but v%d is not a vertex of f%d\n",
                  neighbor->id, vertex->id, vertex->id, neighbor->id);
              errorfacet2= errorfacet;
              errorfacet= neighbor;
            }
          }
        }
        FORALLfacet_(facetlist){
          if (!facet->visible) {
            /* vertices are inverse sorted and are unlikely to be duplicated */
            FOREACHvertex_(facet->vertices){
              if (!qh_setin(vertex->neighbors, facet)) {
                qh_fprintf(qh, qh->ferr, 6277, "qhull internal error (qh_checkpolygon): v%d is a vertex of facet f%d but f%d is not a neighbor of v%d\n",
                  vertex->id, facet->id, facet->id, vertex->id);
                errorfacet2= errorfacet;
                errorfacet= facet;
              }
            }
          }
        }
        waserror= True;
      }
    }
    if (numvertices != qh->num_vertices - qh_setsize(qh, qh->del_vertices)) {
      qh_fprintf(qh, qh->ferr, 6142, "qhull internal error (qh_checkpolygon): actual number of vertices is %d, cumulative vertex count is %d\n",
              numvertices, qh->num_vertices - qh_setsize(qh, qh->del_vertices));
      waserror= True;
    }
    if (qh->hull_dim == 2 && numvertices != numfacets) {
      qh_fprintf(qh, qh->ferr, 6143, "qhull internal error (qh_checkpolygon): #vertices %d != #facets %d\n",
        numvertices, numfacets);
      waserror= True;
    }
    if (qh->hull_dim == 3 && numvertices + numfacets - numridges/2 != 2) {
      qh_fprintf(qh, qh->ferr, 7063, "qhull warning: #vertices %d + #facets %d - #edges %d != 2\n\
        A vertex appears twice in a edge list.  May occur during merging.",
        numvertices, numfacets, numridges/2);
      /* occurs if lots of merging and a vertex ends up twice in an edge list.  e.g., RBOX 1000 s W1e-13 t995849315 D2 | QHULL d Tc Tv */
    }
  }
  if (waserror)
    qh_errexit2(qh, qh_ERRqhull, errorfacet, errorfacet2);
} /* checkpolygon */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkvertex">-</a>

  qh_checkvertex(qh, vertex )
    check vertex for consistency
    checks vertex->neighbors

  notes:
    neighbors checked efficiently in checkpolygon
*/
void qh_checkvertex(qhT *qh, vertexT *vertex) {
  boolT waserror= False;
  facetT *neighbor, **neighborp, *errfacet=NULL;

  if (qh_pointid(qh, vertex->point) == qh_IDunknown) {
    qh_fprintf(qh, qh->ferr, 6144, "qhull internal error (qh_checkvertex): unknown point id %p\n", vertex->point);
    waserror= True;
  }
  if (vertex->id >= qh->vertex_id) {
    qh_fprintf(qh, qh->ferr, 6145, "qhull internal error (qh_checkvertex): unknown vertex id %d\n", vertex->id);
    waserror= True;
  }
  if (!waserror && !vertex->deleted) {
    if (qh_setsize(qh, vertex->neighbors)) {
      FOREACHneighbor_(vertex) {
        if (!qh_setin(neighbor->vertices, vertex)) {
          qh_fprintf(qh, qh->ferr, 6146, "qhull internal error (qh_checkvertex): neighbor f%d does not contain v%d\n", neighbor->id, vertex->id);
          errfacet= neighbor;
          waserror= True;
        }
      }
    }
  }
  if (waserror) {
    qh_errprint(qh, "ERRONEOUS", NULL, NULL, NULL, vertex);
    qh_errexit(qh, qh_ERRqhull, errfacet, NULL);
  }
} /* checkvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="clearcenters">-</a>

  qh_clearcenters(qh, type )
    clear old data from facet->center

  notes:
    sets new centertype
    nop if CENTERtype is the same
*/
void qh_clearcenters(qhT *qh, qh_CENTER type) {
  facetT *facet;

  if (qh->CENTERtype != type) {
    FORALLfacets {
      if (facet->tricoplanar && !facet->keepcentrum)
          facet->center= NULL;  /* center is owned by the ->keepcentrum facet */
      else if (qh->CENTERtype == qh_ASvoronoi){
        if (facet->center) {
          qh_memfree(qh, facet->center, qh->center_size);
          facet->center= NULL;
        }
      }else /* qh->CENTERtype == qh_AScentrum */ {
        if (facet->center) {
          qh_memfree(qh, facet->center, qh->normal_size);
          facet->center= NULL;
        }
      }
    }
    qh->CENTERtype= type;
  }
  trace2((qh, qh->ferr, 2043, "qh_clearcenters: switched to center type %d\n", type));
} /* clearcenters */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="createsimplex">-</a>

  qh_createsimplex(qh, vertices )
    creates a simplex from a set of vertices

  returns:
    initializes qh.facet_list to the simplex
    initializes qh.newfacet_list, .facet_tail
    initializes qh.vertex_list, .newvertex_list, .vertex_tail

  design:
    initializes lists
    for each vertex
      create a new facet
    for each new facet
      create its neighbor set
*/
void qh_createsimplex(qhT *qh, setT *vertices) {
  facetT *facet= NULL, *newfacet;
  boolT toporient= True;
  int vertex_i, vertex_n, nth;
  setT *newfacets= qh_settemp(qh, qh->hull_dim+1);
  vertexT *vertex;

  qh->facet_list= qh->newfacet_list= qh->facet_tail= qh_newfacet(qh);
  qh->num_facets= qh->num_vertices= qh->num_visible= 0;
  qh->vertex_list= qh->newvertex_list= qh->vertex_tail= qh_newvertex(qh, NULL);
  FOREACHvertex_i_(qh, vertices) {
    newfacet= qh_newfacet(qh);
    newfacet->vertices= qh_setnew_delnthsorted(qh, vertices, vertex_n,
                                                vertex_i, 0);
    newfacet->toporient= (unsigned char)toporient;
    qh_appendfacet(qh, newfacet);
    newfacet->newfacet= True;
    qh_appendvertex(qh, vertex);
    qh_setappend(qh, &newfacets, newfacet);
    toporient ^= True;
  }
  FORALLnew_facets {
    nth= 0;
    FORALLfacet_(qh->newfacet_list) {
      if (facet != newfacet)
        SETelem_(newfacet->neighbors, nth++)= facet;
    }
    qh_settruncate(qh, newfacet->neighbors, qh->hull_dim);
  }
  qh_settempfree(qh, &newfacets);
  trace1((qh, qh->ferr, 1028, "qh_createsimplex: created simplex\n"));
} /* createsimplex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="delvertex">-</a>

  qh_delvertex(qh, vertex )
    deletes a vertex and frees its memory

  notes:
    assumes vertex->adjacencies have been updated if needed
    unlinks from vertex_list
*/
void qh_delvertex(qhT *qh, vertexT *vertex) {

  if (vertex->deleted && !vertex->partitioned) {
    qh_fprintf(qh, qh->ferr, 6323, "qhull internal error (qh_delvertex): vertex v%d was deleted but it was not partitioned as a coplanar point\n",
      vertex->id);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  if (vertex == qh->tracevertex)
    qh->tracevertex= NULL;
  qh_removevertex(qh, vertex);
  qh_setfree(qh, &vertex->neighbors);
  qh_memfree(qh, vertex, (int)sizeof(vertexT));
} /* delvertex */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="facet3vertex">-</a>

  qh_facet3vertex(qh, )
    return temporary set of 3-d vertices in qh_ORIENTclock order

  design:
    if simplicial facet
      build set from facet->vertices with facet->toporient
    else
      for each ridge in order
        build set from ridge's vertices
*/
setT *qh_facet3vertex(qhT *qh, facetT *facet) {
  ridgeT *ridge, *firstridge;
  vertexT *vertex;
  int cntvertices, cntprojected=0;
  setT *vertices;

  cntvertices= qh_setsize(qh, facet->vertices);
  vertices= qh_settemp(qh, cntvertices);
  if (facet->simplicial) {
    if (cntvertices != 3) {
      qh_fprintf(qh, qh->ferr, 6147, "qhull internal error (qh_facet3vertex): only %d vertices for simplicial facet f%d\n",
                  cntvertices, facet->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    qh_setappend(qh, &vertices, SETfirst_(facet->vertices));
    if (facet->toporient ^ qh_ORIENTclock)
      qh_setappend(qh, &vertices, SETsecond_(facet->vertices));
    else
      qh_setaddnth(qh, &vertices, 0, SETsecond_(facet->vertices));
    qh_setappend(qh, &vertices, SETelem_(facet->vertices, 2));
  }else {
    ridge= firstridge= SETfirstt_(facet->ridges, ridgeT);   /* no infinite */
    while ((ridge= qh_nextridge3d(ridge, facet, &vertex))) {
      qh_setappend(qh, &vertices, vertex);
      if (++cntprojected > cntvertices || ridge == firstridge)
        break;
    }
    if (!ridge || cntprojected != cntvertices) {
      qh_fprintf(qh, qh->ferr, 6148, "qhull internal error (qh_facet3vertex): ridges for facet %d don't match up.  got at least %d\n",
                  facet->id, cntprojected);
      qh_errexit(qh, qh_ERRqhull, facet, ridge);
    }
  }
  return vertices;
} /* facet3vertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findbestfacet">-</a>

  qh_findbestfacet(qh, point, bestoutside, bestdist, isoutside )
    find facet that is furthest below a point

    for Delaunay triangulations,
      Use qh_setdelaunay() to lift point to paraboloid and scale by 'Qbb' if needed
      Do not use options 'Qbk', 'QBk', or 'QbB' since they scale the coordinates.

  returns:
    if bestoutside is set (e.g., qh_ALL)
      returns best facet that is not upperdelaunay
      if Delaunay and inside, point is outside circumsphere of bestfacet
    else
      returns first facet below point
      if point is inside, returns nearest, !upperdelaunay facet
    distance to facet
    isoutside set if outside of facet

  notes:
    For tricoplanar facets, this finds one of the tricoplanar facets closest
    to the point.  For Delaunay triangulations, the point may be inside a
    different tricoplanar facet. See <a href="../html/qh-code.htm#findfacet">locate a facet with qh_findbestfacet()</a>

    If inside, qh_findbestfacet performs an exhaustive search
       this may be too conservative.  Sometimes it is clearly required.

    qh_findbestfacet is not used by qhull.
    uses qh.visit_id and qh.coplanarset

  see:
    <a href="geom_r.c#findbest">qh_findbest</a>
*/
facetT *qh_findbestfacet(qhT *qh, pointT *point, boolT bestoutside,
           realT *bestdist, boolT *isoutside) {
  facetT *bestfacet= NULL;
  int numpart, totpart= 0;

  bestfacet= qh_findbest(qh, point, qh->facet_list,
                            bestoutside, !qh_ISnewfacets, bestoutside /* qh_NOupper */,
                            bestdist, isoutside, &totpart);
  if (*bestdist < -qh->DISTround) {
    bestfacet= qh_findfacet_all(qh, point, bestdist, isoutside, &numpart);
    totpart += numpart;
    if ((isoutside && *isoutside && bestoutside)
    || (isoutside && !*isoutside && bestfacet->upperdelaunay)) {
      bestfacet= qh_findbest(qh, point, bestfacet,
                            bestoutside, False, bestoutside,
                            bestdist, isoutside, &totpart);
      totpart += numpart;
    }
  }
  trace3((qh, qh->ferr, 3014, "qh_findbestfacet: f%d dist %2.2g isoutside %d totpart %d\n",
          bestfacet->id, *bestdist, (isoutside ? *isoutside : UINT_MAX), totpart));
  return bestfacet;
} /* findbestfacet */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findbestlower">-</a>

  qh_findbestlower(qh, facet, point, bestdist, numpart )
    returns best non-upper, non-flipped neighbor of facet for point
    if needed, searches vertex neighbors

  returns:
    returns bestdist and updates numpart

  notes:
    if Delaunay and inside, point is outside of circumsphere of bestfacet
    called by qh_findbest() for points above an upperdelaunay facet

*/
facetT *qh_findbestlower(qhT *qh, facetT *upperfacet, pointT *point, realT *bestdistp, int *numpart) {
  facetT *neighbor, **neighborp, *bestfacet= NULL;
  realT bestdist= -REALmax/2 /* avoid underflow */;
  realT dist;
  vertexT *vertex;
  boolT isoutside= False;  /* not used */

  zinc_(Zbestlower);
  FOREACHneighbor_(upperfacet) {
    if (neighbor->upperdelaunay || neighbor->flipped)
      continue;
    (*numpart)++;
    qh_distplane(qh, point, neighbor, &dist);
    if (dist > bestdist) {
      bestfacet= neighbor;
      bestdist= dist;
    }
  }
  if (!bestfacet) {
    zinc_(Zbestlowerv);
    /* rarely called, numpart does not count nearvertex computations */
    vertex= qh_nearvertex(qh, upperfacet, point, &dist);
    qh_vertexneighbors(qh);
    FOREACHneighbor_(vertex) {
      if (neighbor->upperdelaunay || neighbor->flipped)
        continue;
      (*numpart)++;
      qh_distplane(qh, point, neighbor, &dist);
      if (dist > bestdist) {
        bestfacet= neighbor;
        bestdist= dist;
      }
    }
  }
  if (!bestfacet) {
    zinc_(Zbestlowerall);  /* invoked once per point in outsideset */
    zmax_(Zbestloweralln, qh->num_facets);
    /* [dec'15] Previously reported as QH6228 */
    trace3((qh, qh->ferr, 3025, "qh_findbestlower: all neighbors of facet %d are flipped or upper Delaunay.  Search all facets\n",
       upperfacet->id));
    /* rarely called */
    bestfacet= qh_findfacet_all(qh, point, &bestdist, &isoutside, numpart);
  }
  *bestdistp= bestdist;
  trace3((qh, qh->ferr, 3015, "qh_findbestlower: f%d dist %2.2g for f%d p%d\n",
          bestfacet->id, bestdist, upperfacet->id, qh_pointid(qh, point)));
  return bestfacet;
} /* findbestlower */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findfacet_all">-</a>

  qh_findfacet_all(qh, point, bestdist, isoutside, numpart )
    exhaustive search for facet below a point

    for Delaunay triangulations,
      Use qh_setdelaunay() to lift point to paraboloid and scale by 'Qbb' if needed
      Do not use options 'Qbk', 'QBk', or 'QbB' since they scale the coordinates.

  returns:
    returns first facet below point
    if point is inside,
      returns nearest facet
    distance to facet
    isoutside if point is outside of the hull
    number of distance tests

  notes:
    primarily for library users, rarely used by Qhull
*/
facetT *qh_findfacet_all(qhT *qh, pointT *point, realT *bestdist, boolT *isoutside,
                          int *numpart) {
  facetT *bestfacet= NULL, *facet;
  realT dist;
  int totpart= 0;

  *bestdist= -REALmax;
  *isoutside= False;
  FORALLfacets {
    if (facet->flipped || !facet->normal)
      continue;
    totpart++;
    qh_distplane(qh, point, facet, &dist);
    if (dist > *bestdist) {
      *bestdist= dist;
      bestfacet= facet;
      if (dist > qh->MINoutside) {
        *isoutside= True;
        break;
      }
    }
  }
  *numpart= totpart;
  trace3((qh, qh->ferr, 3016, "qh_findfacet_all: f%d dist %2.2g isoutside %d totpart %d\n",
          getid_(bestfacet), *bestdist, *isoutside, totpart));
  return bestfacet;
} /* findfacet_all */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findgood">-</a>

  qh_findgood(qh, facetlist, goodhorizon )
    identify good facets for qh.PRINTgood
    if qh.GOODvertex>0
      facet includes point as vertex
      if !match, returns goodhorizon
      inactive if qh.MERGING
    if qh.GOODpoint
      facet is visible or coplanar (>0) or not visible (<0)
    if qh.GOODthreshold
      facet->normal matches threshold
    if !goodhorizon and !match,
      selects facet with closest angle
      sets GOODclosest

  returns:
    number of new, good facets found
    determines facet->good
    may update qh.GOODclosest

  notes:
    called from qh_initbuild, qh_buildcone_onlygood, and qh_findgood_all
    qh_findgood_all (qh_prepare_output) further reduces the good region

  design:
    count good facets
    mark good facets for qh.GOODpoint
    mark good facets for qh.GOODthreshold
    if necessary
      update qh.GOODclosest
*/
int qh_findgood(qhT *qh, facetT *facetlist, int goodhorizon) {
  facetT *facet, *bestfacet= NULL;
  realT angle, bestangle= REALmax, dist;
  int  numgood=0;

  FORALLfacet_(facetlist) {
    if (facet->good)
      numgood++;
  }
  if (qh->GOODvertex>0 && !qh->MERGING) {
    FORALLfacet_(facetlist) {
      if (!qh_isvertex(qh->GOODvertexp, facet->vertices)) {
        facet->good= False;
        numgood--;
      }
    }
  }
  if (qh->GOODpoint && numgood) {
    FORALLfacet_(facetlist) {
      if (facet->good && facet->normal) {
        zinc_(Zdistgood);
        qh_distplane(qh, qh->GOODpointp, facet, &dist);
        if ((qh->GOODpoint > 0) ^ (dist > 0.0)) {
          facet->good= False;
          numgood--;
        }
      }
    }
  }
  if (qh->GOODthreshold && (numgood || goodhorizon || qh->GOODclosest)) {
    FORALLfacet_(facetlist) {
      if (facet->good && facet->normal) {
        if (!qh_inthresholds(qh, facet->normal, &angle)) {
          facet->good= False;
          numgood--;
          if (angle < bestangle) {
            bestangle= angle;
            bestfacet= facet;
          }
        }
      }
    }
    if (!numgood && (!goodhorizon || qh->GOODclosest)) {
      if (qh->GOODclosest) {
        if (qh->GOODclosest->visible)
          qh->GOODclosest= NULL;
        else {
          qh_inthresholds(qh, qh->GOODclosest->normal, &angle);
          if (angle < bestangle)
            bestfacet= qh->GOODclosest;
        }
      }
      if (bestfacet && bestfacet != qh->GOODclosest) {
        if (qh->GOODclosest)
          qh->GOODclosest->good= False;
        qh->GOODclosest= bestfacet;
        bestfacet->good= True;
        numgood++;
        trace2((qh, qh->ferr, 2044, "qh_findgood: f%d is closest(%2.2g) to thresholds\n",
           bestfacet->id, bestangle));
        return numgood;
      }
    }else if (qh->GOODclosest) { /* numgood > 0 */
      qh->GOODclosest->good= False;
      qh->GOODclosest= NULL;
    }
  }
  zadd_(Zgoodfacet, numgood);
  trace2((qh, qh->ferr, 2045, "qh_findgood: found %d good facets with %d good horizon\n",
               numgood, goodhorizon));
  if (!numgood && qh->GOODvertex>0 && !qh->MERGING)
    return goodhorizon;
  return numgood;
} /* findgood */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findgood_all">-</a>

  qh_findgood_all(qh, facetlist )
    apply other constraints for good facets (used by qh.PRINTgood)
    if qh.GOODvertex
      facet includes (>0) or doesn't include (<0) point as vertex
      if last good facet and ONLYgood, prints warning and continues
    if qh.SPLITthresholds
      facet->normal matches threshold, or if none, the closest one
    calls qh_findgood
    nop if good not used

  returns:
    clears facet->good if not good
    sets qh.num_good

  notes:
    this is like qh_findgood but more restrictive

  design:
    uses qh_findgood to mark good facets
    marks facets for qh.GOODvertex
    marks facets for qh.SPLITthreholds
*/
void qh_findgood_all(qhT *qh, facetT *facetlist) {
  facetT *facet, *bestfacet=NULL;
  realT angle, bestangle= REALmax;
  int  numgood=0, startgood;

  if (!qh->GOODvertex && !qh->GOODthreshold && !qh->GOODpoint
  && !qh->SPLITthresholds)
    return;
  if (!qh->ONLYgood)
    qh_findgood(qh, qh->facet_list, 0);
  FORALLfacet_(facetlist) {
    if (facet->good)
      numgood++;
  }
  if (qh->GOODvertex <0 || (qh->GOODvertex > 0 && qh->MERGING)) {
    FORALLfacet_(facetlist) {
      if (facet->good && ((qh->GOODvertex > 0) ^ !!qh_isvertex(qh->GOODvertexp, facet->vertices))) {
        if (!--numgood) {
          if (qh->ONLYgood) {
            qh_fprintf(qh, qh->ferr, 7064, "qhull warning: good vertex p%d does not match last good facet f%d.  Ignored.\n",
               qh_pointid(qh, qh->GOODvertexp), facet->id);
            return;
          }else if (qh->GOODvertex > 0)
            qh_fprintf(qh, qh->ferr, 7065, "qhull warning: point p%d is not a vertex('QV%d').\n",
                qh->GOODvertex-1, qh->GOODvertex-1);
          else
            qh_fprintf(qh, qh->ferr, 7066, "qhull warning: point p%d is a vertex for every facet('QV-%d').\n",
                -qh->GOODvertex - 1, -qh->GOODvertex - 1);
        }
        facet->good= False;
      }
    }
  }
  startgood= numgood;
  if (qh->SPLITthresholds) {
    FORALLfacet_(facetlist) {
      if (facet->good) {
        if (!qh_inthresholds(qh, facet->normal, &angle)) {
          facet->good= False;
          numgood--;
          if (angle < bestangle) {
            bestangle= angle;
            bestfacet= facet;
          }
        }
      }
    }
    if (!numgood && bestfacet) {
      bestfacet->good= True;
      numgood++;
      trace0((qh, qh->ferr, 23, "qh_findgood_all: f%d is closest(%2.2g) to thresholds\n",
           bestfacet->id, bestangle));
      return;
    }
  }
  qh->num_good= numgood;
  trace0((qh, qh->ferr, 24, "qh_findgood_all: %d good facets remain out of %d facets\n",
        numgood, startgood));
} /* findgood_all */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="furthestnext">-</a>

  qh_furthestnext()
    set qh.facet_next to facet with furthest of all furthest points
    searches all facets on qh.facet_list

  notes:
    this may help avoid precision problems
*/
void qh_furthestnext(qhT *qh /* qh->facet_list */) {
  facetT *facet, *bestfacet= NULL;
  realT dist, bestdist= -REALmax;

  FORALLfacets {
    if (facet->outsideset) {
#if qh_COMPUTEfurthest
      pointT *furthest;
      furthest= (pointT*)qh_setlast(facet->outsideset);
      zinc_(Zcomputefurthest);
      qh_distplane(qh, furthest, facet, &dist);
#else
      dist= facet->furthestdist;
#endif
      if (dist > bestdist) {
        bestfacet= facet;
        bestdist= dist;
      }
    }
  }
  if (bestfacet) {
    qh_removefacet(qh, bestfacet);
    qh_prependfacet(qh, bestfacet, &qh->facet_next);
    trace1((qh, qh->ferr, 1029, "qh_furthestnext: made f%d next facet(dist %.2g)\n",
            bestfacet->id, bestdist));
  }
} /* furthestnext */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="furthestout">-</a>

  qh_furthestout(qh, facet )
    make furthest outside point the last point of outsideset

  returns:
    updates facet->outsideset
    clears facet->notfurthest
    sets facet->furthestdist

  design:
    determine best point of outsideset
    make it the last point of outsideset
*/
void qh_furthestout(qhT *qh, facetT *facet) {
  pointT *point, **pointp, *bestpoint= NULL;
  realT dist, bestdist= -REALmax;

  FOREACHpoint_(facet->outsideset) {
    qh_distplane(qh, point, facet, &dist);
    zinc_(Zcomputefurthest);
    if (dist > bestdist) {
      bestpoint= point;
      bestdist= dist;
    }
  }
  if (bestpoint) {
    qh_setdel(facet->outsideset, point);
    qh_setappend(qh, &facet->outsideset, point);
#if !qh_COMPUTEfurthest
    facet->furthestdist= bestdist;
#endif
  }
  facet->notfurthest= False;
  trace3((qh, qh->ferr, 3017, "qh_furthestout: p%d is furthest outside point of f%d\n",
          qh_pointid(qh, point), facet->id));
} /* furthestout */


/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="infiniteloop">-</a>

  qh_infiniteloop(qh, facet )
    report infinite loop error due to facet
*/
void qh_infiniteloop(qhT *qh, facetT *facet) {

  qh_fprintf(qh, qh->ferr, 6149, "qhull internal error (qh_infiniteloop): potential infinite loop detected.  If visible, f.replace. If newfacet, f.samecycle\n");
  qh_errexit(qh, qh_ERRqhull, facet, NULL);
} /* qh_infiniteloop */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="initbuild">-</a>

  qh_initbuild()
    initialize hull and outside sets with point array
    qh.FIRSTpoint/qh.NUMpoints is point array
    if qh.GOODpoint
      adds qh.GOODpoint to initial hull

  returns:
    qh_facetlist with initial hull
    points partioned into outside sets, coplanar sets, or inside
    initializes qh.GOODpointp, qh.GOODvertexp,

  design:
    initialize global variables used during qh_buildhull
    determine precision constants and points with max/min coordinate values
      if qh.SCALElast, scale last coordinate(for 'd')
    build initial simplex
    partition input points into facets of initial simplex
    set up lists
    if qh.ONLYgood
      check consistency
      add qh.GOODvertex if defined
*/
void qh_initbuild(qhT *qh) {
  setT *maxpoints, *vertices;
  facetT *facet;
  int i, numpart;
  realT dist;
  boolT isoutside;

  if (qh->PRINTstatistics) {
    qh_fprintf(qh, qh->ferr, 9350, "qhull %s Statistics: %s | %s\n",  /* FIXUP -- moved from stat_r.c */
      qh_version, qh->rbox_command, qh->qhull_command);
    fflush(qh->ferr);
  }
  qh->furthest_id= qh_IDunknown;
  qh->lastreport= 0;
  qh->facet_id= qh->vertex_id= qh->ridge_id= 0;
  qh->visit_id= qh->vertex_visit= 0;
  qh->maxoutdone= False;

  if (qh->GOODpoint > 0)
    qh->GOODpointp= qh_point(qh, qh->GOODpoint-1);
  else if (qh->GOODpoint < 0)
    qh->GOODpointp= qh_point(qh, -qh->GOODpoint-1);
  if (qh->GOODvertex > 0)
    qh->GOODvertexp= qh_point(qh, qh->GOODvertex-1);
  else if (qh->GOODvertex < 0)
    qh->GOODvertexp= qh_point(qh, -qh->GOODvertex-1);
  if ((qh->GOODpoint
       && (qh->GOODpointp < qh->first_point  /* also catches !GOODpointp */
           || qh->GOODpointp > qh_point(qh, qh->num_points-1)))
  || (qh->GOODvertex
       && (qh->GOODvertexp < qh->first_point  /* also catches !GOODvertexp */
           || qh->GOODvertexp > qh_point(qh, qh->num_points-1)))) {
    qh_fprintf(qh, qh->ferr, 6150, "qhull input error: either QGn or QVn point is > p%d\n",
             qh->num_points-1);
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
  maxpoints= qh_maxmin(qh, qh->first_point, qh->num_points, qh->hull_dim);
  if (qh->SCALElast)
    qh_scalelast(qh, qh->first_point, qh->num_points, qh->hull_dim,
               qh->MINlastcoord, qh->MAXlastcoord, qh->MAXwidth);
  qh_detroundoff(qh);
  if (qh->DELAUNAY && qh->upper_threshold[qh->hull_dim-1] > REALmax/2
                  && qh->lower_threshold[qh->hull_dim-1] < -REALmax/2) {
    for (i=qh_PRINTEND; i--; ) {
      if (qh->PRINTout[i] == qh_PRINTgeom && qh->DROPdim < 0
          && !qh->GOODthreshold && !qh->SPLITthresholds)
        break;  /* in this case, don't set upper_threshold */
    }
    if (i < 0) {
      if (qh->UPPERdelaunay) { /* matches qh.upperdelaunay in qh_setfacetplane */
        qh->lower_threshold[qh->hull_dim-1]= qh->ANGLEround * qh_ZEROdelaunay;
        qh->GOODthreshold= True;
      }else {
        qh->upper_threshold[qh->hull_dim-1]= -qh->ANGLEround * qh_ZEROdelaunay;
        if (!qh->GOODthreshold)
          qh->SPLITthresholds= True; /* build upper-convex hull even if Qg */
          /* qh_initqhull_globals errors if Qg without Pdk/etc. */
      }
    }
  }
  vertices= qh_initialvertices(qh, qh->hull_dim, maxpoints, qh->first_point, qh->num_points);
  qh_initialhull(qh, vertices);  /* initial qh->facet_list */
  qh_partitionall(qh, vertices, qh->first_point, qh->num_points);
  if (qh->PRINToptions1st || qh->TRACElevel || qh->IStracing) {
    if (qh->TRACElevel || qh->IStracing)
      qh_fprintf(qh, qh->ferr, 8103, "\nTrace level %d for %s | %s\n",
         qh->IStracing ? qh->IStracing : qh->TRACElevel, qh->rbox_command, qh->qhull_command);
    qh_fprintf(qh, qh->ferr, 8104, "Options selected for Qhull %s:\n%s\n", qh_version, qh->qhull_options);
  }
  qh_resetlists(qh, False, qh_RESETvisible /*qh.visible_list newvertex_list qh.newfacet_list */);
  qh->facet_next= qh->facet_list;
  qh_furthestnext(qh /* qh->facet_list */);
  if (qh->PREmerge) {
    qh->cos_max= qh->premerge_cos;
    qh->centrum_radius= qh->premerge_centrum; /* overwritten by qh_premerge */
  }
  if (qh->ONLYgood) {
    if (qh->GOODvertex > 0 && qh->MERGING) {
      qh_fprintf(qh, qh->ferr, 6151, "qhull input error: 'Qg QVn' (only good vertex) does not work with merging.\nUse 'QJ' to joggle the input or 'Q0' to turn off merging.\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    if (!(qh->GOODthreshold || qh->GOODpoint
         || (!qh->MERGEexact && !qh->PREmerge && qh->GOODvertexp))) {
      qh_fprintf(qh, qh->ferr, 6152, "qhull input error: 'Qg' (ONLYgood) needs a good threshold('Pd0D0'), a\n\
good point(QGn or QG-n), or a good vertex with 'QJ' or 'Q0' (QVn).\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    if (qh->GOODvertex > 0  && !qh->MERGING  /* matches qh_partitionall */
    && !qh_isvertex(qh->GOODvertexp, vertices)) {
      facet= qh_findbestnew(qh, qh->GOODvertexp, qh->facet_list,
                          &dist, !qh_ALL, &isoutside, &numpart);
      zadd_(Zdistgood, numpart);
      if (!isoutside) {
        qh_fprintf(qh, qh->ferr, 6153, "qhull input error: point for QV%d is inside initial simplex.  It can not be made a vertex.\n",
               qh_pointid(qh, qh->GOODvertexp));
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
      }
      if (!qh_addpoint(qh, qh->GOODvertexp, facet, False)) {
        qh_settempfree(qh, &vertices);
        qh_settempfree(qh, &maxpoints);
        return;
      }
    }
    qh_findgood(qh, qh->facet_list, 0);
  }
  qh_settempfree(qh, &vertices);
  qh_settempfree(qh, &maxpoints);
  trace1((qh, qh->ferr, 1030, "qh_initbuild: initial hull created and points partitioned\n"));
} /* initbuild */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="initialhull">-</a>

  qh_initialhull(qh, vertices )
    constructs the initial hull as a DIM3 simplex of vertices

  design:
    creates a simplex (initializes lists)
    determines orientation of simplex
    sets hyperplanes for facets
    doubles checks orientation (in case of axis-parallel facets with Gaussian elimination)
    checks for flipped facets and qh.NARROWhull
    checks the result
*/
void qh_initialhull(qhT *qh, setT *vertices) {
  facetT *facet, *firstfacet, *neighbor, **neighborp;
  realT angle, minangle= REALmax;
#ifndef qh_NOtrace
  int k;
#endif

  qh_createsimplex(qh, vertices);  /* qh->facet_list */
  qh_resetlists(qh, False, qh_RESETvisible);
  qh->facet_next= qh->facet_list;      /* advance facet when processed */
  qh->interior_point= qh_getcenter(qh, vertices);
  firstfacet= qh->facet_list;
  qh_setfacetplane(qh, firstfacet);
  if (firstfacet->flipped) {
    firstfacet->flipped= False;
    FORALLfacets
      facet->toporient ^= (unsigned char)True;
  }
  FORALLfacets
    qh_setfacetplane(qh, facet); 
  FORALLfacets {
    if (facet->flipped) {/* due to axis-parallel facet */
      trace1((qh, qh->ferr, 1031, "qh_initialhull: initial orientation incorrect.  Correcting all facets\n"));
      FORALLfacets { /* reuse facet, then 'break' */
        facet->flipped= False;
        facet->toporient ^= (unsigned char)True;
        qh_orientoutside(qh, facet);
      }
      break;
    }
  }
  FORALLfacets {
    if (!qh_checkflipped(qh, facet, NULL, !qh_ALL)) {  /* can happen with 'R0.1' */
      if (qh->DELAUNAY && ! qh->ATinfinity) {
        if (qh->UPPERdelaunay)
          qh_fprintf(qh, qh->ferr, 6240, "Qhull precision error: Initial simplex is cocircular or cospherical.  Option 'Qs' searches all points.  Can not compute the upper Delaunay triangulation or upper Voronoi diagram of cocircular/cospherical points.\n");
        else
          qh_fprintf(qh, qh->ferr, 6239, "Qhull precision error: Initial simplex is cocircular or cospherical.  Use option 'Qz' for the Delaunay triangulation or Voronoi diagram of cocircular/cospherical points.  Option 'Qz' adds a point \"at infinity\".  Use option 'Qs' to search all points for the initial simplex.\n");
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
      }
      qh_joggle_restart(qh, "initial simplex is flat");
      qh_fprintf(qh, qh->ferr, 6154, "Qhull precision error: Initial simplex is flat (facet %d is coplanar with the interior point)\n",
                   facet->id);
      qh_errexit(qh, qh_ERRsingular, NULL, NULL);  /* calls qh_printhelp_singular */
    }
    FOREACHneighbor_(facet) {
      angle= qh_getangle(qh, facet->normal, neighbor->normal);
      minimize_( minangle, angle);
    }
  }
  if (minangle < qh_MAXnarrow && !qh->NOnarrow) {
    realT diff= 1.0 + minangle;

    qh->NARROWhull= True;
    qh_option(qh, "_narrow-hull", NULL, &diff);
    if (minangle < qh_WARNnarrow && !qh->RERUN && qh->PRINTprecision)
      qh_printhelp_narrowhull(qh, qh->ferr, minangle);
  }
  zzval_(Zprocessed)= qh->hull_dim+1;
  qh_checkpolygon(qh, qh->facet_list);
  qh_checkconvex(qh, qh->facet_list,   qh_DATAfault);
#ifndef qh_NOtrace
  if (qh->IStracing >= 1) {
    qh_fprintf(qh, qh->ferr, 8105, "qh_initialhull: simplex constructed, interior point:");
    for (k=0; k < qh->hull_dim; k++)
      qh_fprintf(qh, qh->ferr, 8106, " %6.4g", qh->interior_point[k]);
    qh_fprintf(qh, qh->ferr, 8107, "\n");
  }
#endif
} /* initialhull */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="initialvertices">-</a>

  qh_initialvertices(qh, dim, maxpoints, points, numpoints )
    determines a non-singular set of initial vertices
    maxpoints may include duplicate points

  returns:
    temporary set of dim+1 vertices in descending order by vertex id
    if qh.RANDOMoutside && !qh.ALLpoints
      picks random points
    if dim >= qh_INITIALmax,
      uses min/max x and max points with non-zero determinants

  notes:
    unless qh.ALLpoints,
      uses maxpoints as long as determinate is non-zero
*/
setT *qh_initialvertices(qhT *qh, int dim, setT *maxpoints, pointT *points, int numpoints) {
  pointT *point, **pointp;
  setT *vertices, *simplex, *tested;
  realT randr;
  int idx, point_i, point_n, k;
  boolT nearzero= False;

  vertices= qh_settemp(qh, dim + 1);
  simplex= qh_settemp(qh, dim+1);
  if (qh->ALLpoints)
    qh_maxsimplex(qh, dim, NULL, points, numpoints, &simplex);
  else if (qh->RANDOMoutside) {
    while (qh_setsize(qh, simplex) != dim+1) {
      randr= qh_RANDOMint;
      randr= randr/(qh_RANDOMmax+1);
      idx= (int)floor(qh->num_points * randr);
      while (qh_setin(simplex, qh_point(qh, idx))) {
            idx++; /* in case qh_RANDOMint always returns the same value */
        idx= idx < qh->num_points ? idx : 0;
      }
      qh_setappend(qh, &simplex, qh_point(qh, idx));
    }
  }else if (qh->hull_dim >= qh_INITIALmax) {
    tested= qh_settemp(qh, dim+1);
    qh_setappend(qh, &simplex, SETfirst_(maxpoints));   /* max and min X coord */
    qh_setappend(qh, &simplex, SETsecond_(maxpoints));
    qh_maxsimplex(qh, fmin_(qh_INITIALsearch, dim), maxpoints, points, numpoints, &simplex);
    k= qh_setsize(qh, simplex);
    FOREACHpoint_i_(qh, maxpoints) {
      if (point_i & 0x1) {     /* first pick up max. coord. points */
        if (!qh_setin(simplex, point) && !qh_setin(tested, point)){
          qh_detsimplex(qh, point, simplex, k, &nearzero);
          if (nearzero)
            qh_setappend(qh, &tested, point);
          else {
            qh_setappend(qh, &simplex, point);
            if (++k == dim)  /* use search for last point */
              break;
          }
        }
      }
    }
    while (k != dim && (point= (pointT*)qh_setdellast(maxpoints))) {
      if (!qh_setin(simplex, point) && !qh_setin(tested, point)){
        qh_detsimplex(qh, point, simplex, k, &nearzero);
        if (nearzero)
          qh_setappend(qh, &tested, point);
        else {
          qh_setappend(qh, &simplex, point);
          k++;
        }
      }
    }
    idx= 0;
    while (k != dim && (point= qh_point(qh, idx++))) {
      if (!qh_setin(simplex, point) && !qh_setin(tested, point)){
        qh_detsimplex(qh, point, simplex, k, &nearzero);
        if (!nearzero){
          qh_setappend(qh, &simplex, point);
          k++;
        }
      }
    }
    qh_settempfree(qh, &tested);
    qh_maxsimplex(qh, dim, maxpoints, points, numpoints, &simplex);
  }else
    qh_maxsimplex(qh, dim, maxpoints, points, numpoints, &simplex);
  FOREACHpoint_(simplex)
    qh_setaddnth(qh, &vertices, 0, qh_newvertex(qh, point)); /* descending order */
  qh_settempfree(qh, &simplex);
  return vertices;
} /* initialvertices */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="isvertex">-</a>

  qh_isvertex( point, vertices )
    returns vertex if point is in vertex set, else returns NULL

  notes:
    for qh.GOODvertex
*/
vertexT *qh_isvertex(pointT *point, setT *vertices) {
  vertexT *vertex, **vertexp;

  FOREACHvertex_(vertices) {
    if (vertex->point == point)
      return vertex;
  }
  return NULL;
} /* isvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="makenewfacets">-</a>

  qh_makenewfacets(qh, point )
    make new facets from point and qh.visible_list

  returns:
    apex (point) of the new facets
    qh.newfacet_list= list of new facets with hyperplanes and ->newfacet
    qh.newvertex_list= list of vertices in new facets with ->newfacet set

    if (qh.NEWtentative)
      newfacets reference horizon facets, but not vice versa
      ridges reference non-simplicial horizon ridges, but not vice versa
      does not change existing facets
    else
      sets qh.NEWfacets
      new facets attached to horizon facets and ridges
      for visible facets,
        visible->r.replace is corresponding new facet

  see also:
    qh_makenewplanes() -- make hyperplanes for facets
    qh_attachnewfacets() -- attachnewfacets if not done here qh->NEWtentative
    qh_matchnewfacets() -- match up neighbors
    qh_updatevertices() -- update vertex neighbors and delvertices
    qh_deletevisible() -- delete visible facets
    qh_checkpolygon() --check the result
    qh_triangulate() -- triangulate a non-simplicial facet

  design:
    for each visible facet
      make new facets to its horizon facets
      update its f.replace
      clear its neighbor set
*/
vertexT *qh_makenewfacets(qhT *qh, pointT *point /*visible_list*/) {
  facetT *visible, *newfacet= NULL, *newfacet2= NULL, *neighbor, **neighborp;
  vertexT *apex;
  int numnew=0;

  if (qh->CHECKfrequently) {
    qh_checkdelridge(qh);
  }
  qh->newfacet_list= qh->facet_tail;
  qh->newvertex_list= qh->vertex_tail;
  apex= qh_newvertex(qh, point);
  qh_appendvertex(qh, apex);
  qh->visit_id++;
  if (!qh->NEWtentative)
    qh->NEWfacets= True;
  FORALLvisible_facets {
    FOREACHneighbor_(visible)
      neighbor->seen= False;
    if (visible->ridges) { 
      visible->visitid= qh->visit_id;
      newfacet2= qh_makenew_nonsimplicial(qh, visible, apex, &numnew);
    }
    if (visible->simplicial)
      newfacet= qh_makenew_simplicial(qh, visible, apex, &numnew);
    if (!qh->NEWtentative) {
      if (newfacet2)  /* newfacet is null if all ridges defined */
        newfacet= newfacet2;
      if (newfacet)
        visible->f.replace= newfacet;
      else
        zinc_(Zinsidevisible);
      SETfirst_(visible->neighbors)= NULL;
    }
  }
  trace1((qh, qh->ferr, 1032, "qh_makenewfacets: created %d new facets f%d..f%d from point p%d to horizon\n",
    numnew, qh->first_newfacet, qh->facet_id-1, qh_pointid(qh, point)));
  if (qh->IStracing >= 4)
    qh_printfacetlist(qh, qh->newfacet_list, NULL, qh_ALL);
  return apex;
} /* makenewfacets */

#ifndef qh_NOmerge
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="matchdupridge">-</a>

  qh_matchdupridge(qh, atfacet, atskip, hashsize, hashcount )
    match duplicate ridges in qh.hash_table for atfacet@atskip
    duplicates marked with ->dupridge and qh_DUPLICATEridge

  returns:
    vertex-facet distance (>0.0) for qh_MERGEridge ridge
    updates hashcount
    set newfacet, facet, matchfacet's hyperplane (removes from mergecycle of coplanarhorizon facets)

  see also:
    qh_matchneighbor

  notes:
    same embedded loops as for qh_matchdupridge_coplanarhorizon (may hash in any order)
    assumes atfacet->neighbors @ atskip == qh_DUPLICATEridge
    called by qh_matchnewfacets for qh_buildcone and qh_triangulate_facet

  design:
    compute hash value for atfacet and atskip
    repeat twice -- once to make best matches, once to match the rest
      for each possible facet in qh.hash_table
        if it is a matching facet with the same orientation and pass 2
          make match
          unless tricoplanar, mark match for merging (qh_MERGEridge)
          [e.g., tricoplanar RBOX s 1000 t993602376 | QHULL C-1e-3 d Qbb FA Qt]
        if it is a matching facet with the same orientation and pass 1
          test if this is a better match
      if pass 1,
        make best match (it will not be merged)
        set newfacet, facet, matchfacet's hyperplane (removes from mergecycle of coplanarhorizon facets)

*/
coordT qh_matchdupridge(qhT *qh, facetT *atfacet, int atskip, int hashsize, int *hashcount) {
  boolT same, ismatch;
  int hash, scan;
  facetT *facet, *newfacet, *maxmatch= NULL, *maxmatch2= NULL, *nextfacet;
  int skip, newskip, nextskip= 0, maxskip= 0, maxskip2= 0, samecycle_count= 0, makematch;
  coordT maxdist= -REALmax, maxdist2= 0.0, dupdist, dupdist2, low, high;

  hash= qh_gethash(qh, hashsize, atfacet->vertices, qh->hull_dim, 1,
                     SETelem_(atfacet->vertices, atskip));
  trace2((qh, qh->ferr, 2046, "qh_matchdupridge: find duplicate matches for f%d skip %d hash %d hashcount %d\n",
          atfacet->id, atskip, hash, *hashcount));
  for (makematch= 0; makematch < 2; makematch++) { /* makematch is false on the first pass and 1 on the second */
    qh->visit_id++;
    for (newfacet= atfacet, newskip= atskip; newfacet; newfacet= nextfacet, newskip= nextskip) {
      zinc_(Zhashlookup);
      nextfacet= NULL; /* exit when ismatch found */
      newfacet->visitid= qh->visit_id;
      for (scan= hash; (facet= SETelemt_(qh->hash_table, scan, facetT));
           scan= (++scan >= hashsize ? 0 : scan)) {
        if (!facet->dupridge || facet->visitid == qh->visit_id)
          continue;
        zinc_(Zhashtests);
        if (qh_matchvertices(qh, 1, newfacet->vertices, newskip, facet->vertices, &skip, &same)) {
          if (SETelem_(newfacet->vertices, newskip) == SETelem_(facet->vertices, skip)) {
            trace3((qh, qh->ferr, 3053, "qh_matchdupridge: duplicate ridge due to duplicate facets (f%d skip %d and f%d skip %d) previously reported as QH7084.  Maximize dupdist to force vertex merge\n",
              newfacet->id, newskip, facet->id, skip));
            maxdist2= REALmax/2;
          }
          ismatch= (same == (boolT)(newfacet->toporient ^ facet->toporient));
          if (SETelemt_(facet->neighbors, skip, facetT) != qh_DUPLICATEridge) {
            if (!makematch) {
              qh_fprintf(qh, qh->ferr, 6155, "qhull internal error (qh_matchdupridge): missing qh_DUPLICATEridge at f%d skip %d for new f%d skip %d hash %d ismatch %d.  Set by qh_matchneighbor\n",
                facet->id, skip, newfacet->id, newskip, hash, ismatch);
              qh_errexit2(qh, qh_ERRqhull, facet, newfacet);
            }
          }else if (!ismatch) {
            nextfacet= facet;
            nextskip= skip;
          }else if (SETelemt_(newfacet->neighbors, newskip, facetT) == qh_DUPLICATEridge) {
            if (makematch) {
              if (newfacet->tricoplanar) {
                SETelem_(facet->neighbors, skip)= newfacet;
                SETelem_(newfacet->neighbors, newskip)= facet;
                *hashcount -= 2; /* removed two unmatched facets */
                trace2((qh, qh->ferr, 2075, "qh_matchdupridge: allow tricoplanar duplicate ridge for new f%d skip %d and f%d skip %d\n",
                    newfacet->id, newskip, facet->id, skip)); /* FIXUP -- how is tricoplanar duplicate ridge handled? */
              }else {
                SETelem_(facet->neighbors, skip)= newfacet;
                SETelem_(newfacet->neighbors, newskip)= qh_MERGEridge;  /* resolved by qh_mark_dupridges */
                *hashcount -= 2; /* removed two unmatched facets */
                trace4((qh, qh->ferr, 4059, "qh_matchdupridge: need forced merge of duplicate ridge for new f%d skip %d and f%d skip %d in qh_forcedmerges\n",
                  newfacet->id, newskip, facet->id, skip));
              }
            }else { /* !makematch */
              if (!facet->normal)
                qh_setfacetplane(qh, facet); /* qh_mergecycle will ignore 'mergehorizon' facets with normals, too many cases otherwise */
              if (!newfacet->normal) 
                qh_setfacetplane(qh, newfacet);
              dupdist= qh_getdistance(qh, facet, newfacet, &low, &high); /* ignore low/high */
              dupdist2= qh_getdistance(qh, newfacet, facet, &low, &high);
              minimize_(dupdist, dupdist2);
              /* if a facet is flipped, match the closest for merging by qh_flippedmerges */
              if (maxmatch && maxmatch->flipped) {
                /* keep flipped facets for merging by qh_flippedmerges */ 
                if (maxmatch2->flipped) {
                  /* keep flipped, matched facets */
                }else if (facet->flipped && (dupdist < maxdist || newfacet->flipped)) {
                  maxdist= dupdist;
                  maxmatch= facet;
                  maxskip= skip;
                  maxmatch2= newfacet;
                  maxskip2= newskip;
                }else if (newfacet->flipped && dupdist < maxdist) {
                  maxdist= dupdist;
                  maxmatch= newfacet;
                  maxskip= newskip;
                  maxmatch2= facet;
                  maxskip2= skip;
                } 
                if (dupdist != maxdist && dupdist > maxdist2)
                  maxdist2= dupdist;
              }else if (facet->flipped) {
                maxdist2= maxdist;
                maxdist= dupdist;
                maxmatch= facet;
                maxskip= skip;
                maxmatch2= newfacet;
                maxskip2= newskip;
              }else if (newfacet->flipped) {
                maxdist2= maxdist;
                maxdist= dupdist;
                maxmatch= newfacet;
                maxskip= newskip;
                maxmatch2= facet;
                maxskip2= skip;
              }else if (dupdist > maxdist) { /* otherwise match the furthest apart facets */
                maxdist2= maxdist;
                maxdist= dupdist;
                maxmatch= facet;
                maxskip= skip;
                maxmatch2= newfacet;
                maxskip2= newskip;
              }else if (dupdist > maxdist2)
                maxdist2= dupdist;
              if (qh->IStracing >= 3) {
                if (maxmatch) 
                  qh_fprintf(qh, qh->ferr, 3018, "qh_matchdupridge: duplicate ridge for new f%d skip %d and f%d skip %d at dist %2.2g, maxdist %2.2g f%d f%d flipped? %d, maxdist2 %2.2g\n",
                    newfacet->id, newskip, facet->id, skip, dupdist, maxdist, maxmatch->id, maxmatch2->id, maxmatch->flipped, maxdist2);
                else
                  qh_fprintf(qh, qh->ferr, 3055, "qh_matchdupridge: duplicate ridge for new f%d skip %d and f%d skip %d at dist %2.2g\n",
                    newfacet->id, newskip, facet->id, skip, dupdist);
              }
            }
          }
        }
      } /* end of foreach entry in qh.hash_table starting at 'hash' */
      if (makematch && SETelemt_(newfacet->neighbors, newskip, facetT) == qh_DUPLICATEridge) {
        qh_fprintf(qh, qh->ferr, 6156, "qhull internal error (qh_matchdupridge): no MERGEridge match for duplicate ridge for new f%d skip %d at hash %d..%d\n",
                    newfacet->id, newskip, hash, scan);
        qh_errexit(qh, qh_ERRqhull, newfacet, NULL);
      }
    } /* end of foreach newfacet at 'hash' */
    if (!makematch) {
      if (!maxmatch) {
        qh_fprintf(qh, qh->ferr, 6157, "qhull internal error (qh_matchdupridge): no maximum match for duplicate ridge for new f%d skip %d at hash %d..%d\n",
          atfacet->id, atskip, hash, scan);
        qh_errexit(qh, qh_ERRqhull, atfacet, NULL);
      }
      SETelem_(maxmatch->neighbors, maxskip)= maxmatch2; /* maxmatch!=NULL by QH6157 */
      SETelem_(maxmatch2->neighbors, maxskip2)= maxmatch;
      *hashcount -= 2; /* removed two unmatched facets */
      zzinc_(Zmultiridge);
      trace0((qh, qh->ferr, 25, "qh_matchdupridge: keep duplicate ridge for new f%d skip %d and f%d skip %d\n",
        maxmatch2->id, maxskip2, maxmatch->id, maxskip));
      if (qh->IStracing >= 5)
        qh_errprint(qh, "keep one DUPLICATED ridge facet and its MATCH", maxmatch2, maxmatch, NULL, NULL);
    }
  }
  return maxdist2;
} /* matchdupridge */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="matchdupridge_coplanarhorizon">-</a>

  qh_matchdupridge_coplanarhorizon(qh, atfacet, atskip, hashsize, hashcount )
    match duplicate ridges with the same coplanar horizon in qh.hash_table for atfacet@atskip
	alternate to qh_matchdupridge in qh_matchnewfacets
    duplicates marked with f.dupridge and qh_DUPLICATEridge neighbors

  returns:
    updates hashcount -- 0 if all matched

  notes:
    same embedded loops as for qh_matchdupridge (may hash in any order)
    assumes qh_matchneighbor set atfacet->neighbors @ atskip == qh_DUPLICATEridge
    called by qh_matchnewfacets for qh_buildcone*

  design:  FIXUP redo
    compute hash value for atfacet and atskip
    repeat twice -- once to make best matches, once to match the rest
      for each possible facet in qh.hash_table
        if it is a matching facet with the same orientation and pass 2
          make match
          unless tricoplanar, mark match for merging (qh_MERGEridge)
          [e.g., tricoplanar RBOX s 1000 t993602376 | QHULL C-1e-3 d Qbb FA Qt]
        if it is a matching facet with the same orientation and pass 1
          test if this is a better match
      if pass 1,
        make mergehorizon match if two facets share the same coplanar horizon facet
*/
void qh_matchdupridge_coplanarhorizon(qhT *qh, facetT *atfacet, int atskip, int hashsize, int *hashcount) {
  facetT *facet, *newfacet, *nextfacet, *horizon, *newhorizon, *neighbor;
  facetT *horizonmatch= NULL, *horizonnewmatch= NULL, *firstmatch= NULL, *firstnewmatch= NULL; 
  int horizonskip= 0, horizonnewskip= 0, firstskip= 0, firstnewskip= 0;
  int nextskip= 0, samecycle_count= 0;
  int makematch, hash, scan, skip, newskip, neighbor_i, neighbor_n;
  boolT same, ismatch, ismergehorizon, isbesthorizon= False, isfirstmatch= True, ismergeridge= False, isdup;

  hash= qh_gethash(qh, hashsize, atfacet->vertices, qh->hull_dim, 1,
                     SETelem_(atfacet->vertices, atskip));
  trace2((qh, qh->ferr, 2078, "qh_matchdupridge_coplanarhorizon: try to match duplicate ridges with the same coplanar horizon for f%d skip %d hash %d hashcount %d\n",
          atfacet->id, atskip, hash, *hashcount));
  for (makematch= 0; makematch < 2; makematch++) { /* makematch is false (0) on the first pass and true (1) on the second */
    qh->visit_id++;
    for (newfacet= atfacet, newskip= atskip; newfacet; newfacet= nextfacet, newskip= nextskip) {
      zinc_(Zhashlookup);
      nextfacet= NULL; /* exit when ismatch found */
      newfacet->visitid= qh->visit_id;
      for (scan= hash; (facet= SETelemt_(qh->hash_table, scan, facetT));
           scan= (++scan >= hashsize ? 0 : scan)) {
        if (!facet->dupridge || facet->visitid == qh->visit_id)
          continue;
        zinc_(Zhashtests);
        if (qh_matchvertices(qh, 1 /*firstindex*/, newfacet->vertices, newskip, facet->vertices, &skip, &same)) {
          ismatch= (same == (boolT)(newfacet->toporient ^ facet->toporient));
          ismergehorizon= False;  /* True if ridge of mergehorizon facets of the same horizon */
          if (facet->mergehorizon && newfacet->mergehorizon) {
            horizon= SETfirst_(facet->neighbors);
            newhorizon=  SETfirst_(newfacet->neighbors);
            if (horizon == newhorizon) {
              ismergehorizon= True;
            }
          }
          if (SETelemt_(facet->neighbors, skip, facetT) != qh_DUPLICATEridge) {
            if (!makematch) {
              /* FIXUP -- also in qh_matchdupridge.  Who wins?  not chased, occurred for 'rbox 75 C3,2e-13 D4 t1541088956 | qhull d Tcv' */
              if (SETelemt_(facet->neighbors, skip, facetT) == newfacet)
                qh_fprintf(qh, qh->ferr, 6106, "qhull precision error (qh_matchdupridge_coplanarhorizon): missing qh_DUPLICATEridge at f%d skip %d for new f%d skip %d hash %d ismatch %d.  May be due to QH7084 (duplicate vertices)\n",
                  facet->id, skip, newfacet->id, newskip, hash, ismatch);
              else {
                qh_fprintf(qh, qh->ferr, 6302, "qhull topology error (qh_matchdupridge_coplanarhorizon): missing qh_DUPLICATEridge at f%d skip %d for new f%d skip %d hash %d ismatch %d.  Set by qh_matchneighbor\n",
                  facet->id, skip, newfacet->id, newskip, hash, ismatch);
                /* QH6302 is apparently rare -- due to >2 dups (QH2080), all dups (QH2100), and coplanarhorizon/!coplanarhorizon
                   rbox 175 C3,2e-13 D4 t1544542567 | qhull d Tcv */
              }
              qh_errexit2(qh, qh_ERRqhull, facet, newfacet);
            }
          }else if (!ismatch) {
            nextfacet= facet;
            nextskip= skip;
          }else if (SETelemt_(newfacet->neighbors, newskip, facetT) == qh_DUPLICATEridge) {
            if (makematch && ismergehorizon) {
              SETelem_(facet->neighbors, skip)= newfacet;
              SETelem_(newfacet->neighbors, newskip)= facet;
              *hashcount -= 2; /* removed two unmatched facets */
              trace2((qh, qh->ferr, 2079, "qh_matchdupridge_coplanarhorizon: match duplicate ridge with same coplanar horizon f%d -- new f%d skip %d and f%d skip %d\n",
                horizon->id, newfacet->id, newskip, facet->id, skip));
            }else if (makematch && isfirstmatch) {
              isfirstmatch= False;
              firstmatch= facet; /* if other match is mergehorizon, will match firstmatch/firstnewmatch */
              firstskip= skip;
              firstnewmatch= newfacet;
              firstnewskip= newskip;
              /* temporarily mark this match to remove it from consideration */
              SETelem_(facet->neighbors, skip)= qh_MERGEridge;
              SETelem_(newfacet->neighbors, newskip)= qh_MERGEridge;
            }else if (makematch) {
              firstmatch= NULL; /* undo firstmatch, will need qh_matchdupridge */
              /* temporarily mark this match to remove it from consideration */
              SETelem_(facet->neighbors, skip)= qh_MERGEridge;
              SETelem_(newfacet->neighbors, newskip)= qh_MERGEridge;
            }else if (ismergehorizon) {  /* !makematch */
              horizonmatch= facet; /* will match a facet/newfacet with the same coplanar horizon facet */
              horizonskip= skip;
              horizonnewmatch= newfacet;
              horizonnewskip= newskip;
              samecycle_count++;   /* may be double counted, will match the last one */
              trace4((qh, qh->ferr, 4089, "qh_matchdupridge_coplanarhorizon: %d duplicate ridges with coplanarhorizon -- new f%d skip %d and f%d skip %d\n",
                samecycle_count, newfacet->id, newskip, facet->id, skip));
            }
          }
        }
      } /* end of foreach entry in qh.hash_table starting at 'hash' */
      if (makematch && SETelemt_(newfacet->neighbors, newskip, facetT) == qh_DUPLICATEridge) {  /* FIXUP was False &&, review further */
        qh_fprintf(qh, qh->ferr, 6303, "qhull internal error (qh_matchdupridge_coplanarhorizon): no MERGEridge match for duplicate ridge for new f%d skip %d at hash %d..%d\n",
                    newfacet->id, newskip, hash, scan);
        qh_errexit(qh, qh_ERRqhull, newfacet, NULL);
      }
      if (nextfacet) 
        trace4((qh, qh->ferr, 4083, "qh_matchdupridge_coplanarhorizon: test next f%d skip %d\n", nextfacet->id, nextskip));
    } /* end of foreach newfacet at 'hash' */
    if (!makematch) {
      if (horizonmatch) { 
        SETelem_(horizonmatch->neighbors, horizonskip)= horizonnewmatch;
        SETelem_(horizonnewmatch->neighbors, horizonnewskip)= horizonmatch;
        *hashcount -= 2; /* removed two unmatched facets */
        trace0((qh, qh->ferr, 29, "qh_matchdupridge_coplanarhorizon: keep a duplicate ridge with coplanar horizon new f%d skip %d and f%d skip %d\n",
          horizonnewmatch->id, horizonnewskip, horizonmatch->id, horizonskip));
        if (qh->IStracing >= 5)
          qh_errprint(qh, "keep one mergehorizon DUPLICATED ridge MATCH", horizonnewmatch, horizonmatch, NULL, NULL);
      }
    }
  }
  if (firstmatch) {
    SETelem_(firstmatch->neighbors, firstskip)= firstnewmatch;
    SETelem_(firstnewmatch->neighbors, firstnewskip)= firstmatch;
    *hashcount -= 2; /* removed two unmatched facets */
    zzinc_(Zmultiridge);  /* FIXUP, stats, drop trace0? */
    trace0((qh, qh->ferr, 28, "qh_matchdupridge_coplanarhorizon: keep first ridge -- new f%d skip %d and f%d skip %d\n",
      firstnewmatch->id, firstnewskip, firstmatch->id, firstskip));
    if (qh->IStracing >= 5)
      qh_errprint(qh, "keep first ridge MATCH", firstnewmatch, firstmatch, NULL, NULL);
  }else if (isfirstmatch) {
    trace2((qh, qh->ferr, 2100, "qh_matchdupridge_coplanarhorizon: all duplicate ridges will merge into a coplanar horizon samecycle_count %d hashcount %d\n",
      samecycle_count, *hashcount));
  }else {
    trace2((qh, qh->ferr, 2080, "qh_matchdupridge_coplanarhorizon: duplicate ridges need merging -- hashcount %d, samecycle_count %d \n",
      *hashcount, samecycle_count));
  }
  FORALLnew_facets {  /* restore qh_DUPLICATEridge for unmatched neighbors */
    if (newfacet->dupridge) {
      isdup= False;
      FOREACHneighbor_i_(qh, newfacet) {
        if (neighbor == qh_MERGEridge) {
          SETelem_(newfacet->neighbors, neighbor_i)= qh_DUPLICATEridge;
          isdup= True;
        }else if (neighbor == qh_DUPLICATEridge)
          isdup= True;
      }
      if (!isdup)
        newfacet->dupridge= False;
    }
  }
} /* matchdupridge_coplanarhorizon */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="nearcoplanar">-</a>

  qh_nearcoplanar()
    for all facets, remove near-inside points from facet->coplanarset</li>
    coplanar points defined by innerplane from qh_outerinner()

  returns:
    if qh->KEEPcoplanar && !qh->KEEPinside
      facet->coplanarset only contains coplanar points
    if qh.JOGGLEmax
      drops inner plane by another qh.JOGGLEmax diagonal since a
        vertex could shift out while a coplanar point shifts in

  notes:
    used for qh.PREmerge and qh.JOGGLEmax
    must agree with computation of qh.NEARcoplanar in qh_detroundoff(qh)
  design:
    if not keeping coplanar or inside points
      free all coplanar sets
    else if not keeping both coplanar and inside points
      remove !coplanar or !inside points from coplanar sets
*/
void qh_nearcoplanar(qhT *qh /* qh.facet_list */) {
  facetT *facet;
  pointT *point, **pointp;
  int numpart;
  realT dist, innerplane;

  if (!qh->KEEPcoplanar && !qh->KEEPinside) {
    FORALLfacets {
      if (facet->coplanarset)
        qh_setfree(qh, &facet->coplanarset);
    }
  }else if (!qh->KEEPcoplanar || !qh->KEEPinside) {
    qh_outerinner(qh, NULL, NULL, &innerplane);
    if (qh->JOGGLEmax < REALmax/2)
      innerplane -= qh->JOGGLEmax * sqrt((realT)qh->hull_dim);
    numpart= 0;
    FORALLfacets {
      if (facet->coplanarset) {
        FOREACHpoint_(facet->coplanarset) {
          numpart++;
          qh_distplane(qh, point, facet, &dist);
          if (dist < innerplane) {
            if (!qh->KEEPinside)
              SETref_(point)= NULL;
          }else if (!qh->KEEPcoplanar)
            SETref_(point)= NULL;
        }
        qh_setcompact(qh, facet->coplanarset);
      }
    }
    zzadd_(Zcheckpart, numpart);
  }
} /* nearcoplanar */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="nearvertex">-</a>

  qh_nearvertex(qh, facet, point, bestdist )
    return nearest vertex in facet to point

  returns:
    vertex and its distance

  notes:
    if qh.DELAUNAY
      distance is measured in the input set
    searches neighboring tricoplanar facets (requires vertexneighbors)
      Slow implementation.  Recomputes vertex set for each point.
    The vertex set could be stored in the qh.keepcentrum facet.
*/
vertexT *qh_nearvertex(qhT *qh, facetT *facet, pointT *point, realT *bestdistp) {
  realT bestdist= REALmax, dist;
  vertexT *bestvertex= NULL, *vertex, **vertexp, *apex;
  coordT *center;
  facetT *neighbor, **neighborp;
  setT *vertices;
  int dim= qh->hull_dim;

  if (qh->DELAUNAY)
    dim--;
  if (facet->tricoplanar) {
    if (!qh->VERTEXneighbors || !facet->center) {
      qh_fprintf(qh, qh->ferr, 6158, "qhull internal error (qh_nearvertex): qh.VERTEXneighbors and facet->center required for tricoplanar facets\n");
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    vertices= qh_settemp(qh, qh->TEMPsize);
    apex= SETfirstt_(facet->vertices, vertexT);
    center= facet->center;
    FOREACHneighbor_(apex) {
      if (neighbor->center == center) {
        FOREACHvertex_(neighbor->vertices)
          qh_setappend(qh, &vertices, vertex);
      }
    }
  }else
    vertices= facet->vertices;
  FOREACHvertex_(vertices) {
    dist= qh_pointdist(vertex->point, point, -dim);
    if (dist < bestdist) {
      bestdist= dist;
      bestvertex= vertex;
    }
  }
  if (facet->tricoplanar)
    qh_settempfree(qh, &vertices);
  *bestdistp= sqrt(bestdist);
  if (!bestvertex) {
      qh_fprintf(qh, qh->ferr, 6261, "qhull internal error (qh_nearvertex): did not find bestvertex for f%d p%d\n", facet->id, qh_pointid(qh, point));
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }
  trace3((qh, qh->ferr, 3019, "qh_nearvertex: v%d dist %2.2g for f%d p%d\n",
        bestvertex->id, *bestdistp, facet->id, qh_pointid(qh, point))); /* bestvertex!=0 by QH2161 */
  return bestvertex;
} /* nearvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="newhashtable">-</a>

  qh_newhashtable(qh, newsize )
    returns size of qh.hash_table of at least newsize slots

  notes:
    assumes qh.hash_table is NULL
    qh_HASHfactor determines the number of extra slots
    size is not divisible by 2, 3, or 5
*/
int qh_newhashtable(qhT *qh, int newsize) {
  int size;

  size= ((newsize+1)*qh_HASHfactor) | 0x1;  /* odd number */
  while (True) {
    if (newsize<0 || size<0) {
        qh_fprintf(qh, qh->qhmem.ferr, 6236, "qhull error (qh_newhashtable): negative request (%d) or size (%d).  Did int overflow due to high-D?\n", newsize, size); /* WARN64 */
        qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
    }
    if ((size%3) && (size%5))
      break;
    size += 2;
    /* loop terminates because there is an infinite number of primes */
  }
  qh->hash_table= qh_setnew(qh, size);
  qh_setzero(qh, qh->hash_table, 0, size);
  return size;
} /* newhashtable */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="newvertex">-</a>

  qh_newvertex(qh, point )
    returns a new vertex for point
*/
vertexT *qh_newvertex(qhT *qh, pointT *point) {
  vertexT *vertex;

  zinc_(Ztotvertices);
  vertex= (vertexT *)qh_memalloc(qh, (int)sizeof(vertexT));
  memset((char *) vertex, (size_t)0, sizeof(vertexT));
  if (qh->vertex_id == UINT_MAX) {
    qh_memfree(qh, vertex, (int)sizeof(vertexT));
    qh_fprintf(qh, qh->ferr, 6159, "qhull error: more than 2^32 vertices.  vertexT.id field overflows.  Vertices would not be sorted correctly.\n");
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  if (qh->vertex_id == qh->tracevertex_id)
    qh->tracevertex= vertex;
  vertex->id= qh->vertex_id++;
  vertex->point= point;
  trace4((qh, qh->ferr, 4060, "qh_newvertex: vertex p%d(v%d) created\n", qh_pointid(qh, vertex->point),
          vertex->id));
  return(vertex);
} /* newvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="nextridge3d">-</a>

  qh_nextridge3d( atridge, facet, vertex )
    return next ridge and vertex for a 3d facet
    returns NULL on error
    [for QhullFacet::nextRidge3d] Does not call qh_errexit nor access qhT.

  notes:
    in qh_ORIENTclock order
    this is a O(n^2) implementation to trace all ridges
    be sure to stop on any 2nd visit
    same as QhullRidge::nextRidge3d
    does not use qhT or qh_errexit [QhullFacet.cpp]

  design:
    for each ridge
      exit if it is the ridge after atridge
*/
ridgeT *qh_nextridge3d(ridgeT *atridge, facetT *facet, vertexT **vertexp) {
  vertexT *atvertex, *vertex, *othervertex;
  ridgeT *ridge, **ridgep;

  if ((atridge->top == facet) ^ qh_ORIENTclock)
    atvertex= SETsecondt_(atridge->vertices, vertexT);
  else
    atvertex= SETfirstt_(atridge->vertices, vertexT);
  FOREACHridge_(facet->ridges) {
    if (ridge == atridge)
      continue;
    if ((ridge->top == facet) ^ qh_ORIENTclock) {
      othervertex= SETsecondt_(ridge->vertices, vertexT);
      vertex= SETfirstt_(ridge->vertices, vertexT);
    }else {
      vertex= SETsecondt_(ridge->vertices, vertexT);
      othervertex= SETfirstt_(ridge->vertices, vertexT);
    }
    if (vertex == atvertex) {
      if (vertexp)
        *vertexp= othervertex;
      return ridge;
    }
  }
  return NULL;
} /* nextridge3d */
#else /* qh_NOmerge */
coordT qh_matchdupridge(qhT *qh, facetT *atfacet, int atskip, int hashsize, int *hashcount) {
  return 0.0;
}
void qh_matchdupridge_coplanarhorizon(qhT *qh, facetT *atfacet, int atskip, int hashsize, int *hashcount) {
}
ridgeT *qh_nextridge3d(ridgeT *atridge, facetT *facet, vertexT **vertexp) {
  return NULL;
}
#endif /* qh_NOmerge */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="opposite_vertex">-</a>

  qh_opposite_vertex(qh, facetA, neighbor)
    return the opposite vertex in facetA to neighbor

*/
vertexT *qh_opposite_vertex(qhT *qh, facetT *facetA,  facetT *neighbor) {
    vertexT *opposite= NULL;
    facetT *facet;
    int facet_i, facet_n;

    if (facetA->simplicial) {
      FOREACHfacet_i_(qh, facetA->neighbors) {
        if (facet == neighbor) {
          opposite= SETelem_(facetA->vertices, facet_i);
          break;
        }
      }
    }
    if (!opposite) {
      qh_fprintf(qh, qh->ferr, 6324, "qhull internal error (qh_opposite_vertex): opposite vertex in facet f%d to neighbor f%d is not defined.  Either is facet is not simplicial or neighbor not found\n",
        facet->id, neighbor->id);
      qh_errexit2(qh, qh_ERRqhull, facet, neighbor);
    }
    return opposite;
} /* opposite_vertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="outcoplanar">-</a>

  qh_outcoplanar()
    move points from all facets' outsidesets to their coplanarsets

  notes:
    for post-processing under qh.NARROWhull

  design:
    for each facet
      for each outside point for facet
        partition point into coplanar set
*/
void qh_outcoplanar(qhT *qh /* facet_list */) {
  pointT *point, **pointp;
  facetT *facet;
  realT dist;

  trace1((qh, qh->ferr, 1033, "qh_outcoplanar: move outsideset to coplanarset for qh->NARROWhull\n"));
  FORALLfacets {
    FOREACHpoint_(facet->outsideset) {
      qh->num_outside--;
      if (qh->KEEPcoplanar || qh->KEEPnearinside) {
        qh_distplane(qh, point, facet, &dist);
        zinc_(Zpartition);
        qh_partitioncoplanar(qh, point, facet, &dist, qh->findbestnew);
      }
    }
    qh_setfree(qh, &facet->outsideset);
  }
} /* outcoplanar */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="point">-</a>

  qh_point(qh, id )
    return point for a point id, or NULL if unknown

  alternative code:
    return((pointT *)((unsigned   long)qh.first_point
           + (unsigned long)((id)*qh.normal_size)));
*/
pointT *qh_point(qhT *qh, int id) {

  if (id < 0)
    return NULL;
  if (id < qh->num_points)
    return qh->first_point + id * qh->hull_dim;
  id -= qh->num_points;
  if (id < qh_setsize(qh, qh->other_points))
    return SETelemt_(qh->other_points, id, pointT);
  return NULL;
} /* point */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="point_add">-</a>

  qh_point_add(qh, set, point, elem )
    stores elem at set[point.id]

  returns:
    access function for qh_pointfacet and qh_pointvertex

  notes:
    checks point.id
*/
void qh_point_add(qhT *qh, setT *set, pointT *point, void *elem) {
  int id, size;

  SETreturnsize_(set, size);
  if ((id= qh_pointid(qh, point)) < 0)
    qh_fprintf(qh, qh->ferr, 7067, "qhull internal warning (point_add): unknown point %p id %d\n",
      point, id);
  else if (id >= size) {
    qh_fprintf(qh, qh->ferr, 6160, "qhull internal errror(point_add): point p%d is out of bounds(%d)\n",
             id, size);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }else
    SETelem_(set, id)= elem;
} /* point_add */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="pointfacet">-</a>

  qh_pointfacet()
    return temporary set of facet for each point
    the set is indexed by point id
    at most one facet per point, arbitrary selection

  notes:
    each point is assigned to at most one of vertices, coplanarset, or outsideset
    unassigned points are interior points or 
    vertices assigned to one of its facets
    coplanarset assigned to the facet
    outside set assigned to the facet
    NULL if no facet for point (inside)
      includes qh.GOODpointp


  access:
    FOREACHfacet_i_(qh, facets) { ... }
    SETelem_(facets, i)

  design:
    for each facet
      add each vertex
      add each coplanar point
      add each outside point
*/
setT *qh_pointfacet(qhT *qh /*qh.facet_list*/) {
  int numpoints= qh->num_points + qh_setsize(qh, qh->other_points);
  setT *facets;
  facetT *facet;
  vertexT *vertex, **vertexp;
  pointT *point, **pointp;

  facets= qh_settemp(qh, numpoints);
  qh_setzero(qh, facets, 0, numpoints);
  qh->vertex_visit++;
  FORALLfacets {
    FOREACHvertex_(facet->vertices) {
      if (vertex->visitid != qh->vertex_visit) {
        vertex->visitid= qh->vertex_visit;
        qh_point_add(qh, facets, vertex->point, facet);
      }
    }
    FOREACHpoint_(facet->coplanarset)
      qh_point_add(qh, facets, point, facet);
    FOREACHpoint_(facet->outsideset)
      qh_point_add(qh, facets, point, facet);
  }
  return facets;
} /* pointfacet */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="pointvertex">-</a>

  qh_pointvertex(qh, )
    return temporary set of vertices indexed by point id
    entry is NULL if no vertex for a point
      this will include qh.GOODpointp

  access:
    FOREACHvertex_i_(qh, vertices) { ... }
    SETelem_(vertices, i)
*/
setT *qh_pointvertex(qhT *qh /*qh.facet_list*/) {
  int numpoints= qh->num_points + qh_setsize(qh, qh->other_points);
  setT *vertices;
  vertexT *vertex;

  vertices= qh_settemp(qh, numpoints);
  qh_setzero(qh, vertices, 0, numpoints);
  FORALLvertices
    qh_point_add(qh, vertices, vertex->point, vertex);
  return vertices;
} /* pointvertex */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="prependfacet">-</a>

  qh_prependfacet(qh, facet, facetlist )
    prepend facet to the start of a facetlist

  returns:
    increments qh.numfacets
    updates facetlist, qh.facet_list, facet_next

  notes:
    be careful of prepending since it can lose a pointer.
      e.g., can lose _next by deleting and then prepending before _next
*/
void qh_prependfacet(qhT *qh, facetT *facet, facetT **facetlist) {
  facetT *prevfacet, *list;


  trace4((qh, qh->ferr, 4061, "qh_prependfacet: prepend f%d before f%d\n",
          facet->id, getid_(*facetlist)));
  if (!*facetlist)
    (*facetlist)= qh->facet_tail;
  list= *facetlist;
  prevfacet= list->previous;
  facet->previous= prevfacet;
  if (prevfacet)
    prevfacet->next= facet;
  list->previous= facet;
  facet->next= *facetlist;
  if (qh->facet_list == list)  /* this may change *facetlist */
    qh->facet_list= facet;
  if (qh->facet_next == list)
    qh->facet_next= facet;
  *facetlist= facet;
  qh->num_facets++;
} /* prependfacet */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="printhashtable">-</a>

  qh_printhashtable(qh, fp )
    print hash table to fp

  notes:
    not in I/O to avoid bringing io_r.c in

  design:
    for each hash entry
      if defined
        if unmatched or will merge (NULL, qh_MERGEridge, qh_DUPLICATEridge)
          print entry and neighbors
*/
void qh_printhashtable(qhT *qh, FILE *fp) {
  facetT *facet, *neighbor;
  int id, facet_i, facet_n, neighbor_i= 0, neighbor_n= 0;
  vertexT *vertex, **vertexp;

  FOREACHfacet_i_(qh, qh->hash_table) {
    if (facet) {
      FOREACHneighbor_i_(qh, facet) {
        if (!neighbor || neighbor == qh_MERGEridge || neighbor == qh_DUPLICATEridge)
          break;
      }
      if (neighbor_i == neighbor_n)
        continue;
      qh_fprintf(qh, fp, 9283, "hash %d f%d ", facet_i, facet->id);
      FOREACHvertex_(facet->vertices)
        qh_fprintf(qh, fp, 9284, "v%d ", vertex->id);
      qh_fprintf(qh, fp, 9285, "\n neighbors:");
      FOREACHneighbor_i_(qh, facet) {
        if (neighbor == qh_MERGEridge)
          id= -3;
        else if (neighbor == qh_DUPLICATEridge)
          id= -2;
        else
          id= getid_(neighbor);
        qh_fprintf(qh, fp, 9286, " %d", id);
      }
      qh_fprintf(qh, fp, 9287, "\n");
    }
  }
} /* printhashtable */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="addfacetvertex">-</a>

  qh_replacefacetvertex( qh, facet, oldvertex, newvertex )
    replace oldvertex with newvertex in f.vertices
    vertices are inverse sorted by vertex->id

  returns:
    toporient is flipped if an odd parity, position change

  notes:
    for simplicial facets in qh_rename_adjacentvertex
    see qh_addfacetvertex
*/
void qh_replacefacetvertex(qhT *qh, facetT *facet, vertexT *oldvertex, vertexT *newvertex) {
  vertexT *vertex;
  facetT *neighbor;
  int vertex_i, vertex_n;
  int old_i= -1, new_i= -1;

  trace3((qh, qh->ferr, 3038, "qh_replacefacetvertex: replace v%d with v%d in f%d\n", oldvertex->id, newvertex->id, facet->id));
  if (!facet->simplicial) {
    qh_fprintf(qh, qh->ferr, 6283, "qhull internal error (qh_replacefacetvertex): f%d is not simplicial\n", facet->id);
    qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }
  FOREACHvertex_i_(qh, facet->vertices) {
    if (new_i == -1 && vertex->id < newvertex->id) {
      new_i= vertex_i;
    }else if (vertex->id == newvertex->id) {
      qh_fprintf(qh, qh->ferr, 6281, "qhull internal error (qh_replacefacetvertex): f%d already contains new v%d\n", facet->id, newvertex->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    if (vertex->id == oldvertex->id) {
      old_i= vertex_i;
    }
  }
  if (old_i == -1) {
    qh_fprintf(qh, qh->ferr, 6282, "qhull internal error (qh_replacefacetvertex): f%d does not contain old v%d\n", facet->id, oldvertex->id);
    qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }
  if (new_i == -1) {
    new_i= vertex_n;
  }
  if (old_i < new_i)
    new_i--;
  if ((old_i & 0x1) != (new_i & 0x1))
    facet->toporient ^= 1;
  qh_setdelnthsorted(qh, facet->vertices, old_i);
  qh_setaddnth(qh, &facet->vertices, new_i, newvertex);
  neighbor= SETelem_(facet->neighbors, old_i);
  qh_setdelnthsorted(qh, facet->neighbors, old_i);
  qh_setaddnth(qh, &facet->neighbors, new_i, neighbor);
} /* replacefacetvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="resetlists">-</a>

  qh_resetlists(qh, stats, qh_RESETvisible )
    reset newvertex_list, newfacet_list, visible_list, NEWfacets, NEWtentative
    if stats,
      maintains statistics
    if resetVisible, 
      visible_list is restored to facet_list

  returns:
    newvertex_list, newfacet_list, visible_list are NULL

  notes:
    To delete visible facets, call qh_deletevisible before qh_resetlists
*/
void qh_resetlists(qhT *qh, boolT stats, boolT resetVisible /*qh.newvertex_list newfacet_list visible_list*/) {
  vertexT *vertex;
  facetT *newfacet, *visible;
  int totnew=0, totver=0;

  trace2((qh, qh->ferr, 2066, "qh_resetlists: reset newvertex_list v%d, newfacet_list f%d, visible_list f%d, facet_list f%d next f%d vertex_list v%d -- NEWfacets? %d, NEWtentative? %d, stats? %d\n",
    getid_(qh->newvertex_list), getid_(qh->newfacet_list), getid_(qh->visible_list), getid_(qh->facet_list), getid_(qh->facet_next), getid_(qh->vertex_list), qh->NEWfacets, qh->NEWtentative, stats));
  if (stats) {
    FORALLvertex_(qh->newvertex_list)
      totver++;
    FORALLnew_facets
      totnew++;
    zadd_(Zvisvertextot, totver);
    zmax_(Zvisvertexmax, totver);
    zadd_(Znewfacettot, totnew);
    zmax_(Znewfacetmax, totnew);
  }
  FORALLvertex_(qh->newvertex_list)
    vertex->newfacet= False;
  qh->newvertex_list= NULL;
  FORALLnew_facets {
    newfacet->newfacet= False;
    newfacet->dupridge= False;
  }
  qh->newfacet_list= NULL;
  if (resetVisible) {
    FORALLvisible_facets {
      visible->f.replace= NULL;
      visible->visible= False;
    }
    qh->num_visible= 0;
  }
  qh->visible_list= NULL; /* may still have visible facets via qh_triangulate */
  qh->NEWfacets= False;
  qh->NEWtentative= False;
} /* resetlists */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="setvoronoi_all">-</a>

  qh_setvoronoi_all(qh)
    compute Voronoi centers for all facets
    includes upperDelaunay facets if qh.UPPERdelaunay ('Qu')

  returns:
    facet->center is the Voronoi center

  notes:
    this is unused/untested code
      please email bradb@shore.net if this works ok for you

  use:
    FORALLvertices {...} to locate the vertex for a point.
    FOREACHneighbor_(vertex) {...} to visit the Voronoi centers for a Voronoi cell.
*/
void qh_setvoronoi_all(qhT *qh) {
  facetT *facet;

  qh_clearcenters(qh, qh_ASvoronoi);
  qh_vertexneighbors(qh);

  FORALLfacets {
    if (!facet->normal || !facet->upperdelaunay || qh->UPPERdelaunay) {
      if (!facet->center)
        facet->center= qh_facetcenter(qh, facet->vertices);
    }
  }
} /* setvoronoi_all */

#ifndef qh_NOmerge
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate">-</a>

  qh_triangulate()
    triangulate non-simplicial facets on qh.facet_list,
    if qh->VORONOI, sets Voronoi centers of non-simplicial facets
    nop if hasTriangulation

  returns:
    all facets simplicial
    each tricoplanar facet has ->f.triowner == owner of ->center,normal,etc.

  notes:
    call after qh_check_output since may switch to Voronoi centers
    Output may overwrite ->f.triowner with ->f.area
    See qh_buildcone
*/
void qh_triangulate(qhT *qh /*qh.facet_list*/) {
  facetT *facet, *nextfacet, *owner;
  int onlygood= qh->ONLYgood;
  facetT *neighbor, *visible= NULL, *facet1, *facet2, *new_facet_list= NULL;
  facetT *orig_neighbor= NULL, *otherfacet;
  vertexT *new_vertex_list= NULL;
  mergeT *merge;
  mergeType mergetype;
  int neighbor_i, neighbor_n;

  if (qh->hasTriangulation)
      return;
  trace1((qh, qh->ferr, 1034, "qh_triangulate: triangulate non-simplicial facets\n"));
  if (qh->hull_dim == 2)
    return;
  if (qh->VORONOI) {  /* otherwise lose Voronoi centers [could rebuild vertex set from tricoplanar] */
    qh_clearcenters(qh, qh_ASvoronoi);
    qh_vertexneighbors(qh);
  }
  qh->ONLYgood= False; /* for makenew_nonsimplicial */
  qh->visit_id++;
  qh->NEWfacets= True;
  qh_initmergesets(qh, qh_ALL);
  qh->newvertex_list= qh->vertex_tail;
  for (facet= qh->facet_list; facet && facet->next; facet= nextfacet) { /* non-simplicial facets moved to end */
    nextfacet= facet->next;
    if (facet->visible || facet->simplicial)
      continue;
    /* triangulate all non-simplicial facets, otherwise merging does not work, e.g., RBOX c P-0.1 P+0.1 P+0.1 D3 | QHULL d Qt Tv */
    if (!new_facet_list)
      new_facet_list= facet;  /* will be moved to end */
    qh_triangulate_facet(qh, facet, &new_vertex_list);
  }
  trace2((qh, qh->ferr, 2047, "qh_triangulate: delete null facets from facetlist f%d.  A null facet has the same first (apex) and second vertices\n", getid_(new_facet_list)));
  for (facet= new_facet_list; facet && facet->next; facet= nextfacet) { /* null facets moved to end */
    nextfacet= facet->next;
    if (facet->visible)
      continue;
    if (facet->ridges) {
      if (qh_setsize(qh, facet->ridges) > 0) {
        qh_fprintf(qh, qh->ferr, 6161, "qhull error (qh_triangulate): ridges still defined for f%d\n", facet->id);
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
      }
      qh_setfree(qh, &facet->ridges);
    }
    if (SETfirst_(facet->vertices) == SETsecond_(facet->vertices)) {
      zinc_(Ztrinull);
      qh_triangulate_null(qh, facet);
    }
  }
  trace2((qh, qh->ferr, 2048, "qh_triangulate: delete %d or more mirrored facets.  Mirrored facets have the same vertices due to a null facet\n", qh_setsize(qh, qh->degen_mergeset)));
  qh->visible_list= qh->facet_tail;
  while ((merge= (mergeT*)qh_setdellast(qh->degen_mergeset))) {
    facet1= merge->facet1;
    facet2= merge->facet2;
    mergetype= merge->mergetype;
    qh_memfree(qh, merge, (int)sizeof(mergeT));
    if (mergetype == MRGmirror) {
      zinc_(Ztrimirror);
      qh_triangulate_mirror(qh, facet1, facet2);
    }
  }
  qh_freemergesets(qh, qh_ALL);
  trace2((qh, qh->ferr, 2049, "qh_triangulate: update neighbor lists for vertices from v%d\n", getid_(new_vertex_list)));
  qh->newvertex_list= new_vertex_list;  /* all vertices of new facets */
  qh->visible_list= NULL;
  qh_updatevertices(qh /*qh.newvertex_list, empty newfacet_list and visible_list*/);
  qh_resetlists(qh, False, !qh_RESETvisible /*qh.newvertex_list, empty newfacet_list and visible_list*/);

  trace2((qh, qh->ferr, 2050, "qh_triangulate: identify degenerate tricoplanar facets from f%d\n", getid_(new_facet_list)));
  trace2((qh, qh->ferr, 2051, "qh_triangulate: and replace facet->f.triowner with tricoplanar facets that own center, normal, etc.\n"));
  FORALLfacet_(new_facet_list) {
    if (facet->tricoplanar && !facet->visible) {
      FOREACHneighbor_i_(qh, facet) {
        if (neighbor_i == 0) {  /* first iteration */
          if (neighbor->tricoplanar)
            orig_neighbor= neighbor->f.triowner;
          else
            orig_neighbor= neighbor;
        }else {
          if (neighbor->tricoplanar)
            otherfacet= neighbor->f.triowner;
          else
            otherfacet= neighbor;
          if (orig_neighbor == otherfacet) {
            zinc_(Ztridegen);
            facet->degenerate= True;
            break;
          }
        }
      }
    }
  }

  trace2((qh, qh->ferr, 2052, "qh_triangulate: delete visible facets -- non-simplicial, null, and mirrored facets\n"));
  owner= NULL;
  visible= NULL;
  for (facet= new_facet_list; facet && facet->next; facet= nextfacet) { /* may delete facet */
    nextfacet= facet->next;
    if (facet->visible) {
      if (facet->tricoplanar) { /* a null or mirrored facet */
        qh_delfacet(qh, facet);
        qh->num_visible--;
      }else {  /* a non-simplicial facet followed by its tricoplanars */
        if (visible && !owner) {
          /*  RBOX 200 s D5 t1001471447 | QHULL Qt C-0.01 Qx Qc Tv Qt -- f4483 had 6 vertices/neighbors and 8 ridges */
          trace2((qh, qh->ferr, 2053, "qh_triangulate: delete f%d.  All tricoplanar facets degenerate for non-simplicial facet\n",
                       visible->id));
          qh_delfacet(qh, visible);
          qh->num_visible--;
        }
        visible= facet;
        owner= NULL;
      }
    }else if (facet->tricoplanar) {
      if (facet->f.triowner != visible || visible==NULL) {
        qh_fprintf(qh, qh->ferr, 6162, "qhull error (qh_triangulate): tricoplanar facet f%d not owned by its visible, non-simplicial facet f%d\n", facet->id, getid_(visible));
        qh_errexit2(qh, qh_ERRqhull, facet, visible);
      }
      if (owner)
        facet->f.triowner= owner;
      else if (!facet->degenerate) {
        owner= facet;
        nextfacet= visible->next; /* rescan tricoplanar facets with owner, visible!=0 by QH6162 */
        facet->keepcentrum= True;  /* one facet owns ->normal, etc. */
        facet->coplanarset= visible->coplanarset;
        facet->outsideset= visible->outsideset;
        visible->coplanarset= NULL;
        visible->outsideset= NULL;
        if (!qh->TRInormals) { /* center and normal copied to tricoplanar facets */
          visible->center= NULL;
          visible->normal= NULL;
        }
        qh_delfacet(qh, visible);
        qh->num_visible--;
      }
    }
  }
  if (visible && !owner) {
    trace2((qh, qh->ferr, 2054, "qh_triangulate: all tricoplanar facets degenerate for last non-simplicial facet f%d\n",
                 visible->id));
    qh_delfacet(qh, visible);
    qh->num_visible--;
  }
  FORALLfacet_(new_facet_list) {
    facet->degenerate= False; /* reset 'degenerate' flags from last qh_triangulate_facet */
  }
  qh->ONLYgood= onlygood; /* restore value */
  if (qh->CHECKfrequently)
    qh_checkpolygon(qh, qh->facet_list);
  qh->hasTriangulation= True;
} /* triangulate */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate_facet">-</a>

  qh_triangulate_facet(qh, facetA, &firstVertex )
    triangulate a non-simplicial facet
      if qh.CENTERtype=qh_ASvoronoi, sets its Voronoi center
  returns:
    qh.newfacet_list == simplicial facets
      facet->tricoplanar set and ->keepcentrum false
      facet->degenerate set if duplicated apex
      facet->f.trivisible set to facetA
      facet->center copied from facetA (created if qh_ASvoronoi)
        qh_eachvoronoi, qh_detvridge, qh_detvridge3 assume centers copied
      facet->normal,offset,maxoutside copied from facetA

  notes:
      only called by qh_triangulate
      qh_makenew_nonsimplicial uses neighbor->seen for the same
      if qh.TRInormals, newfacet->normal will need qh_free
        if qh.TRInormals and qh_AScentrum, newfacet->center will need qh_free
        keepcentrum is also set on Zwidefacet in qh_mergefacet
        freed by qh_clearcenters

  see also:
      qh_addpoint() -- add a point
      qh_makenewfacets() -- construct a cone of facets for a new vertex

  design:
      if qh_ASvoronoi,
         compute Voronoi center (facet->center)
      select first vertex (highest ID to preserve ID ordering of ->vertices)
      triangulate from vertex to ridges
      copy facet->center, normal, offset
      update vertex neighbors
*/
void qh_triangulate_facet(qhT *qh, facetT *facetA, vertexT **first_vertex) {
  facetT *newfacet;
  facetT *neighbor, **neighborp;
  vertexT *apex;
  int numnew=0;

  trace3((qh, qh->ferr, 3020, "qh_triangulate_facet: triangulate facet f%d\n", facetA->id));

  if (qh->IStracing >= 4)
    qh_printfacet(qh, qh->ferr, facetA);
  FOREACHneighbor_(facetA) {
    neighbor->seen= False;
    neighbor->coplanar= False;
  }
  if (qh->CENTERtype == qh_ASvoronoi && !facetA->center  /* matches upperdelaunay in qh_setfacetplane() */
  && fabs_(facetA->normal[qh->hull_dim -1]) >= qh->ANGLEround * qh_ZEROdelaunay) {
    facetA->center= qh_facetcenter(qh, facetA->vertices);
  }
  qh_willdelete(qh, facetA, NULL);
  qh->newfacet_list= qh->facet_tail;
  facetA->visitid= qh->visit_id;
  apex= SETfirstt_(facetA->vertices, vertexT);
  qh_makenew_nonsimplicial(qh, facetA, apex, &numnew);
  SETfirst_(facetA->neighbors)= NULL;
  FORALLnew_facets {
    newfacet->tricoplanar= True;
    newfacet->f.trivisible= facetA;
    newfacet->degenerate= False;
    newfacet->upperdelaunay= facetA->upperdelaunay;
    newfacet->good= facetA->good;
    if (qh->TRInormals) { /* 'Q11' triangulate duplicates ->normal and ->center */
      newfacet->keepcentrum= True;
      if(facetA->normal){
        newfacet->normal= (double*)qh_memalloc(qh, qh->normal_size);
        memcpy((char *)newfacet->normal, facetA->normal, qh->normal_size);
      }
      if (qh->CENTERtype == qh_AScentrum)
        newfacet->center= qh_getcentrum(qh, newfacet);
      else if (qh->CENTERtype == qh_ASvoronoi && facetA->center){
        newfacet->center= (double*)qh_memalloc(qh, qh->center_size);
        memcpy((char *)newfacet->center, facetA->center, qh->center_size);
      }
    }else {
      newfacet->keepcentrum= False;
      /* one facet will have keepcentrum=True at end of qh_triangulate */
      newfacet->normal= facetA->normal;
      newfacet->center= facetA->center;
    }
    newfacet->offset= facetA->offset;
#if qh_MAXoutside
    newfacet->maxoutside= facetA->maxoutside;
#endif
  }
  qh_matchnewfacets(qh /*qh.newfacet_list*/); /* ignore returned value, maxdupdist */ 
  zinc_(Ztricoplanar);
  zadd_(Ztricoplanartot, numnew);
  zmax_(Ztricoplanarmax, numnew);
  qh->visible_list= NULL;
  if (!(*first_vertex))
    (*first_vertex)= qh->newvertex_list;
  qh->newvertex_list= NULL;
  qh_updatevertices(qh /*qh.newfacet_list, qh.empty visible_list and qh.newvertex_list*/);
  qh_resetlists(qh, False, !qh_RESETvisible /*qh.newfacet_list, qh.empty visible_list and qh.newvertex_list*/);
} /* triangulate_facet */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate_link">-</a>

  qh_triangulate_link(qh, oldfacetA, facetA, oldfacetB, facetB)
    relink facetA to facetB via null oldfacetA or mirrored oldfacetA and oldfacetB
  returns:
    if neighbors are already linked, will merge as MRGmirror (qh.degen_mergeset, 4-d and up)
*/
void qh_triangulate_link(qhT *qh, facetT *oldfacetA, facetT *facetA, facetT *oldfacetB, facetT *facetB) {
  int errmirror= False;

  if (oldfacetA == oldfacetB) {
    trace3((qh, qh->ferr, 3052, "qh_triangulate_link: relink neighbors f%d and f%d of null facet f%d\n",
      facetA->id, facetB->id, oldfacetA->id));
  }else {
    trace3((qh, qh->ferr, 3021, "qh_triangulate_link: relink neighbors f%d and f%d of mirrored facets f%d and f%d\n",
      facetA->id, facetB->id, oldfacetA->id, oldfacetB->id));
  }
  if (qh_setin(facetA->neighbors, facetB)) {
    if (!qh_setin(facetB->neighbors, facetA))
      errmirror= True;
    else if (!qh_hasmerge(qh, qh->degen_mergeset, MRGmirror, facetA, facetB))
      qh_appendmergeset(qh, facetA, facetB, MRGmirror, 0.0, 1.0);
  }else if (qh_setin(facetB->neighbors, facetA))
    errmirror= True;
  if (errmirror) {
    qh_fprintf(qh, qh->ferr, 6163, "qhull error (qh_triangulate_link): neighbors f%d and f%d do not match for null facet or mirrored facets f%d and f%d\n",
       facetA->id, facetB->id, oldfacetA->id, oldfacetB->id);
    qh_errexit2(qh, qh_ERRqhull, facetA, facetB);
  }
  qh_setreplace(qh, facetB->neighbors, oldfacetB, facetA);
  qh_setreplace(qh, facetA->neighbors, oldfacetA, facetB);
} /* triangulate_link */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate_mirror">-</a>

  qh_triangulate_mirror(qh, facetA, facetB)
    delete two mirrored facets identified by qh_triangulate_null() and itself
      a mirrored facet shares the same vertices of a logical ridge
  design:
    since a null facet duplicates the first two vertices, the opposing neighbors absorb the null facet
    if they are already neighbors, the opposing neighbors become MRGmirror facets
*/
void qh_triangulate_mirror(qhT *qh, facetT *facetA, facetT *facetB) {
  facetT *neighbor, *neighborB;
  int neighbor_i, neighbor_n;

  trace3((qh, qh->ferr, 3022, "qh_triangulate_mirror: delete mirrored facets f%d and f%d and link their neighbors\n",
         facetA->id, facetB->id));
  FOREACHneighbor_i_(qh, facetA) {
    neighborB= SETelemt_(facetB->neighbors, neighbor_i, facetT);
    if (neighbor == facetB && neighborB == facetA)
      continue; /* occurs twice */
    else if (neighbor->redundant && neighborB->redundant) { /* also mirrored facets (D5+) */
      if (qh_hasmerge(qh, qh->degen_mergeset, MRGmirror, neighbor, neighborB))
        continue;
    }
    if (neighbor->visible && neighborB->visible) /* previously deleted as mirrored facets */
      continue;
    qh_triangulate_link(qh, facetA, neighbor, facetB, neighborB);
  }
  qh_willdelete(qh, facetA, NULL);
  qh_willdelete(qh, facetB, NULL);
} /* triangulate_mirror */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate_null">-</a>

  qh_triangulate_null(qh, facetA)
    remove null facetA from qh_triangulate_facet()
      a null facet has vertex #1 (apex) == vertex #2
  returns:
    adds facetA to ->visible for deletion after qh_updatevertices
    qh->degen_mergeset contains mirror facets (4-d and up only)
  design:
    since a null facet duplicates the first two vertices, the opposing neighbors absorb the null facet
    if they are already neighbors, the opposing neighbors will be merged (MRGmirror)
*/
void qh_triangulate_null(qhT *qh, facetT *facetA) {
  facetT *neighbor, *otherfacet;

  trace3((qh, qh->ferr, 3023, "qh_triangulate_null: delete null facet f%d\n", facetA->id));
  neighbor= SETfirstt_(facetA->neighbors, facetT);
  otherfacet= SETsecondt_(facetA->neighbors, facetT);
  qh_triangulate_link(qh, facetA, neighbor, facetA, otherfacet);
  qh_willdelete(qh, facetA, NULL);
} /* triangulate_null */

#else /* qh_NOmerge */
void qh_triangulate(qhT *qh) {
}
#endif /* qh_NOmerge */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="vertexintersect">-</a>

  qh_vertexintersect(qh, vertexsetA, vertexsetB )
    intersects two vertex sets (inverse id ordered)
    vertexsetA is a temporary set at the top of qh->qhmem.tempstack

  returns:
    replaces vertexsetA with the intersection

  notes:
    could overwrite vertexsetA if currently too slow
*/
void qh_vertexintersect(qhT *qh, setT **vertexsetA,setT *vertexsetB) {
  setT *intersection;

  intersection= qh_vertexintersect_new(qh, *vertexsetA, vertexsetB);
  qh_settempfree(qh, vertexsetA);
  *vertexsetA= intersection;
  qh_settemppush(qh, intersection);
} /* vertexintersect */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="vertexintersect_new">-</a>

  qh_vertexintersect_new(qh, )
    intersects two vertex sets (inverse id ordered)

  returns:
    a new set
*/
setT *qh_vertexintersect_new(qhT *qh, setT *vertexsetA,setT *vertexsetB) {
  setT *intersection= qh_setnew(qh, qh->hull_dim - 1);
  vertexT **vertexA= SETaddr_(vertexsetA, vertexT);
  vertexT **vertexB= SETaddr_(vertexsetB, vertexT);

  while (*vertexA && *vertexB) {
    if (*vertexA  == *vertexB) {
      qh_setappend(qh, &intersection, *vertexA);
      vertexA++; vertexB++;
    }else {
      if ((*vertexA)->id > (*vertexB)->id)
        vertexA++;
      else
        vertexB++;
    }
  }
  return intersection;
} /* vertexintersect_new */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="vertexneighbors">-</a>

  qh_vertexneighbors(qh)
    for each vertex in qh.facet_list,
      determine its neighboring facets

  returns:
    sets qh.VERTEXneighbors
      nop if qh.VERTEXneighbors already set
      qh_addpoint() will maintain them

  notes:
    assumes all vertex->neighbors are NULL

  design:
    for each facet
      for each vertex
        append facet to vertex->neighbors
*/
void qh_vertexneighbors(qhT *qh /*qh.facet_list*/) {
  facetT *facet;
  vertexT *vertex, **vertexp;

  if (qh->VERTEXneighbors)
    return;
  trace1((qh, qh->ferr, 1035, "qh_vertexneighbors: determining neighboring facets for each vertex\n"));
  qh->vertex_visit++;
  FORALLfacets {
    if (facet->visible)
      continue;
    FOREACHvertex_(facet->vertices) {
      if (vertex->visitid != qh->vertex_visit) {
        vertex->visitid= qh->vertex_visit;
        vertex->neighbors= qh_setnew(qh, qh->hull_dim);
      }
      qh_setappend(qh, &vertex->neighbors, facet);
    }
  }
  qh->VERTEXneighbors= True;
} /* vertexneighbors */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="vertexsubset">-</a>

  qh_vertexsubset( vertexsetA, vertexsetB )
    returns True if vertexsetA is a subset of vertexsetB
    assumes vertexsets are sorted

  note:
    empty set is a subset of any other set
*/
boolT qh_vertexsubset(setT *vertexsetA, setT *vertexsetB) {
  vertexT **vertexA= (vertexT **) SETaddr_(vertexsetA, vertexT);
  vertexT **vertexB= (vertexT **) SETaddr_(vertexsetB, vertexT);

  while (True) {
    if (!*vertexA)
      return True;
    if (!*vertexB)
      return False;
    if ((*vertexA)->id > (*vertexB)->id)
      return False;
    if (*vertexA  == *vertexB)
      vertexA++;
    vertexB++;
  }
  return False; /* avoid warnings */
} /* vertexsubset */
