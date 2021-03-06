% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{gamma_exponential_mixture_bound}
\alias{gamma_exponential_mixture_bound}
\title{Gamma-exponential uniform boundary.}
\usage{
gamma_exponential_mixture_bound(v, alpha, v_opt, c, alpha_opt = 0.05)
}
\arguments{
\item{v}{intrinsic time value}

\item{alpha}{the error / crossing probability of the boundary}

\item{v_opt}{intrinsic time value for which the boundary is optimized}

\item{c}{sub-exponential scale parameter}

\item{alpha_opt}{alpha for which the boundary is optimized}
}
\description{
Gamma-exponential uniform boundary.
}
\examples{
gamma_exponential_mixture_bound(c(100, 200), .05, 100, 2)
}
