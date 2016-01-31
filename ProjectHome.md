It is the code of the following paper:
http://books.nips.cc/papers/files/nips20/NIPS2007_0435.pdf. This is an all-kernel-support version of SVM, which can parallel run on multiple machines.

We migrate it from Google's large scale computing infrastructure to MPI, then every one can use and run it. Please notice this open source project is a 20% project (we do it in part time), and it is still in a Beta version. :)

If you wish to publish any work based on psvm, please cite our paper as:
Edward Chang, Kaihua Zhu, Hao Wang, Hongjie Bai, Jian Li, Zhihuan Qiu, and Hang Cui, PSVM: Parallelizing Support Vector Machines on Distributed Computers. NIPS 2007. Software available at http://code.google.com/p/psvm.

The bibtex format is
```
@InProceedings{psvm,
  author =   {Edward Y. Chang and Kaihua Zhu and Hao Wang and Hongjie Bai and Jian Li and Zhihuan Qiu and Hang Cui},
  title =    {PSVM: Parallelizing Support Vector Machines on Distributed Computers},
  booktitle =    {NIPS},
  year =     {2007},
  note = {Software available at \url{http://code.google.com/p/psvm}}
}

```

If you have any question, please feel free to contact us. And you can also ask your questions on: http://groups.google.com/group/psvm?lnk=srg

Acknowledgment:

We would like to thank National Science Foundation for their grant IIS-0535085, which made the start of this project at UCSB in 2006 possible.