% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{normal_mixture_bound}
\alias{normal_mixture_bound}
\title{One- or two-sided normal mixture uniform boundary.}
\usage{
normal_mixture_bound(v, alpha, v_opt, alpha_opt = 0.05,
  is_one_sided = TRUE)
}
\arguments{
\item{v}{intrinsic time value}

\item{alpha}{the error / crossing probability of the boundary}

\item{v_opt}{intrinsic time value for which the boundary is optimized}

\item{alpha_opt}{alpha for which the boundary is optimized}

\item{is_one_sided}{if FALSE, use the two-sided normal mixture}
}
\description{
One- or two-sided normal mixture uniform boundary.
}
\examples{
normal_mixture_bound(c(100, 200), .05, 100)
normal_mixture_bound(100, c(.05, .01), 100)
}
