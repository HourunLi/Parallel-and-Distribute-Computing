
#include<assert.h>
#include<math.h>
#include<sys/time.h>
#include<stdio.h>
//#include "matrix_storage.h"
#include <stdlib.h>
#include <string.h>
#include <vector>

#include<iostream>
using namespace std;


#define VEC2DWIDTH 1024
#define CSR_VEC_GROUP_SIZE 64
//#define WARPSIZE 64
#define WARPSIZE 32



#define CSR_VEC_MIN_TH_NUM 5760
#define MAX_LEVELS  1000

template <class dimType>
struct matrixInfo
{
    /** Matrix width*/
    dimType width;
     /** Matrix height*/
    dimType height;
    /** Number of non zeros*/
    dimType nnz;

};


template <class dimType, class dataType>
struct coo_matrix
{
    matrixInfo<dimType> matinfo;

    /** Row index, size nnz*/
    dimType* coo_row_id;
    /** Column index, size nnz*/
    dimType* coo_col_id;
    /** Data, size nnz */
    dataType* coo_data;
};

template <class dimType, class dataType>
struct csr_matrix
{
    matrixInfo<dimType> matinfo;

    /** Row pointer, size height + 1*/
    dimType* csr_row_ptr;
    /** Column index, size nnz*/
    dimType* csr_col_id;
    /** Data, size nnz */
    dataType* csr_data;
};

 //extern int findPaddedSize(int realSize, int alignment);

 //extern double distance(float* vec1, float* vec2, int size);

 //extern void two_vec_compare(int* coovec, int* newvec, int size);

 extern double timestamp ();
 
template <class dimType>
 extern void init_mat_info(matrixInfo<dimType>& info);


template <class dimType, class dataType>
 extern bool if_sorted_coo(coo_matrix<dimType, dataType>* mat);


template <class dimType, class dataType>
 extern bool sort_coo(coo_matrix<dimType, dataType>* mat);

//Read from matrix market format to a coo mat
//template <class dimType, class dataType>
 extern void ReadMMF(char* filename, coo_matrix<int, float>* mat);


template <class dimType, class dataType>
extern void init_coo_matrix(coo_matrix<dimType, dataType>& mat);


template <class dimType, class dataType>
 extern void free_coo_matrix(coo_matrix<dimType, dataType>& mat);


//void printMatInfo(coo_matrix<int, float>* mat);


template <class dimType, class dataType>
void coo2csr(coo_matrix<dimType, dataType>* source, csr_matrix<dimType, dataType>* dest);


template<class dimType, class dataType>
void initVectorZero(dataType* vec, dimType vec_size);

template<class dimType, class dataType>
void initVectorOne(dataType* vec, dimType vec_size);


template <class dimType, class dataType>
void coo_spmv(coo_matrix<dimType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size);


//void spmv_only(coo_matrix<int, float>* mat, float* vec, float* coores);


template <class dimType, class dataType>
void init_coo_matrix(coo_matrix<dimType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.coo_row_id = NULL;
    mat.coo_col_id = NULL;
    mat.coo_data = NULL;
}

template <class dimType>
void init_mat_info(matrixInfo<dimType>& info)
{
    info.width = (dimType)0;
    info.height = (dimType)0;
    info.nnz = (dimType)0;
}


template <class dimType, class dataType>
void coo2csr(coo_matrix<dimType, dataType>* source, csr_matrix<dimType, dataType>* dest)
{
    if (!if_sorted_coo(source))
    {
	assert(sort_coo(source) == true);
    }

    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dimType nnz = source->matinfo.nnz;
    dest->csr_row_ptr = (dimType*)malloc(sizeof(dimType)*(source->matinfo.height + 1));
    dest->csr_col_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->csr_data = (dataType*)malloc(sizeof(dataType)*nnz);

    memcpy(dest->csr_data, source->coo_data, sizeof(dataType)*nnz);
    memcpy(dest->csr_col_id, source->coo_col_id, sizeof(dimType)*nnz);

    dest->csr_row_ptr[0] = 0;
    dimType row = (dimType) 0;
    dimType curRow = (dimType) 0;
    while (row < nnz)
    {
	while (source->coo_row_id[row] == curRow && row < nnz)
	    row++;
	curRow++;
	dest->csr_row_ptr[curRow] = row;
    }
    if (curRow < source->matinfo.height)
    {
	curRow++;
	while (curRow <= source->matinfo.height)
	{
	    dest->csr_row_ptr[curRow] = dest->csr_row_ptr[curRow - 1];
	    curRow++;
	}
    }
}


template <class dimType, class dataType>
void change2tran(csr_matrix<dimType, dataType>* source, csr_matrix<dimType, dataType>* dest)
{
    dimType m=source->matinfo.height;
    dimType n=source->matinfo.width;
    dimType nnzL = 0;
    
    if(m<=n)
        n=m;
    else
        m=n;
    
    dest->matinfo.width = m;
    dest->matinfo.height = m;
    

    dimType nnz = source->matinfo.nnz;
    
    
    dest->csr_row_ptr = (dimType*)malloc(sizeof(dimType)*(source->matinfo.height + 1));
    dest->csr_col_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->csr_data = (dataType*)malloc(sizeof(dataType)*nnz);
    
//    memcpy(dest->csr_data, source->coo_data, sizeof(dataType)*nnz);
//    memcpy(dest->csr_col_id, source->coo_col_id, sizeof(dimType)*nnz);
    
    dimType i,j,k,tmp_col,nnz_pointer = 0;;
    dataType tmp_value;
    dest->csr_row_ptr[0] = 0;
    
    for (i = 0; i < m; i++)
    {
        for (j = source->csr_row_ptr[i]; j < source->csr_row_ptr[i+1]; j++)
        {
            tmp_col=source->csr_col_id[j];
            tmp_value=source->csr_data[j];
            for(k=j+1;k<source->csr_row_ptr[i+1];k++)
            {
                if(source->csr_col_id[k]<tmp_col)
                {
                    source->csr_col_id[j]=source->csr_col_id[k];
                    source->csr_data[j]=source->csr_data[k];
                    source->csr_col_id[k]=tmp_col;
                    source->csr_data[k]=tmp_value;
                    tmp_col=source->csr_col_id[j];
                    tmp_value=source->csr_data[j];
                }
            }
            
            if (source->csr_col_id[j] < i)
            {
                dest->csr_col_id[nnz_pointer] = source->csr_col_id[j];
                dest->csr_data[nnz_pointer] = 1;//source->csr_data[j];
                nnz_pointer++;
            }
            else
            {
                break;
            }
        }
        
        dest->csr_col_id[nnz_pointer] = i;
        dest->csr_data[nnz_pointer] = 1.0;
        nnz_pointer++;
        
        dest->csr_row_ptr[i+1] = nnz_pointer;
    }
    
    nnzL = dest->csr_row_ptr[m];
    dest->matinfo.nnz = nnzL;
    
    dest->csr_col_id = (dimType *)realloc(dest->csr_col_id, sizeof(dimType) * nnzL);
    dest->csr_data = (dataType *)realloc(dest->csr_data, sizeof(dataType) * nnzL);
    
}


template <class dimType, class dataType>
void get_x_b(csr_matrix<dimType, dataType>* mat, dataType** x_add, dataType** b_add)
{
    dimType m=mat->matinfo.height;
    dataType *x_ref = (dataType *)malloc(sizeof(dataType) * m);
    int i,j;
    for ( i = 0; i < m; i++)
        x_ref[i] = 1;
    
    dataType *b = (dataType *)malloc(sizeof(dataType) * m);
    
    // run spmv to get b
    for (i = 0; i < m; i++)
    {
        b[i] = 0;
        for (j = mat->csr_row_ptr[i]; j < mat->csr_row_ptr[i+1]; j++)
            b[i] += mat->csr_data[j] * x_ref[mat->csr_col_id[j]];
    }
    
    *x_add=x_ref;
    *b_add=b;
}



template<class dimType, class dataType>
void initVectorZero(dataType* vec, dimType vec_size)
{
    for (dimType i = 0; i < vec_size; i++)
    {
	vec[i] = (dataType)0;
    }
}

template<class dimType, class dataType>
void initVectorOne(dataType* vec, dimType vec_size)
{
    for (dimType i = 0; i < vec_size; i++)
    {
	vec[i] = (dataType)1;
    }
}

template <class dimType, class dataType>
void free_coo_matrix(coo_matrix<dimType, dataType>& mat)
{
    if (mat.coo_row_id != NULL)
	free(mat.coo_row_id);
    if (mat.coo_col_id != NULL)
	free(mat.coo_col_id);
    if (mat.coo_data != NULL)
	free(mat.coo_data);
}

template <class dimType, class dataType>
bool if_sorted_coo(coo_matrix<dimType, dataType>* mat)
{
    dimType nnz = mat->matinfo.nnz;
    for (int i = 0; i < nnz - 1; i++)
    {
        if ((mat->coo_row_id[i] > mat->coo_row_id[i+1]) || (mat->coo_row_id[i] == mat->coo_row_id[i+1] && mat->coo_col_id[i] > mat->coo_col_id[i+1]))
            return false;
    }
    return true;
}



template <class dimType, class dataType>
bool sort_coo(coo_matrix<dimType, dataType>* mat)
{

    int i = 0;
    dimType  beg[MAX_LEVELS], end[MAX_LEVELS], L, R ;
    dimType pivrow, pivcol;
    dataType pivdata;

    beg[0]=0; 
    end[0]=mat->matinfo.nnz;
    while (i>=0) 
    {
	L=beg[i];
	if (end[i] - 1 > end[i])
	    R = end[i];
	else
	    R = end[i] - 1;
	if (L<R) 
	{
	    dimType middle = ((long long int)L+(long long int)R)/2;
	    //dimType middle = (L+R)/2;
	    pivrow=mat->coo_row_id[middle]; 
	    pivcol=mat->coo_col_id[middle];
	    pivdata=mat->coo_data[middle];
	    mat->coo_row_id[middle] = mat->coo_row_id[L];
	    mat->coo_col_id[middle] = mat->coo_col_id[L];
	    mat->coo_data[middle] = mat->coo_data[L];
	    mat->coo_row_id[L] = pivrow;
	    mat->coo_col_id[L] = pivcol;
	    mat->coo_data[L] = pivdata;
	    if (i==MAX_LEVELS-1) 
		return false;
	    while (L<R) 
	    {
		while (((mat->coo_row_id[R] > pivrow) || 
			    (mat->coo_row_id[R] == pivrow && mat->coo_col_id[R] > pivcol)) 
			&& L<R) 
		    R--; 
		if (L<R) 
		{
		    mat->coo_row_id[L] = mat->coo_row_id[R];
		    mat->coo_col_id[L] = mat->coo_col_id[R];
		    mat->coo_data[L] = mat->coo_data[R];
		    L++;
		}
		while (((mat->coo_row_id[L] < pivrow) || 
			    (mat->coo_row_id[L] == pivrow && mat->coo_col_id[L] < pivcol)) 
			&& L<R) 
		    L++; 
		if (L<R) 
		{
		    mat->coo_row_id[R] = mat->coo_row_id[L];
		    mat->coo_col_id[R] = mat->coo_col_id[L];
		    mat->coo_data[R] = mat->coo_data[L];
		    R--;
		}
	    }
	    mat->coo_row_id[L] = pivrow;
	    mat->coo_col_id[L] = pivcol;
	    mat->coo_data[L] = pivdata;
	    beg[i+1]=L+1; 
	    end[i+1]=end[i]; 
	    end[i++]=L; 
	}
	else 
	{
	    i--; 
	}
    }

    return true;
}

//extern void pad_csr(csr_matrix<int, float>* source, csr_matrix<int, float>* dest, int alignment);

template <class dimType, class dataType>
void free_csr_matrix(csr_matrix<dimType, dataType>& mat)
{
    if (mat.csr_row_ptr != NULL)
	free(mat.csr_row_ptr);
    if (mat.csr_col_id != NULL)
	free(mat.csr_col_id);
    if (mat.csr_data != NULL)
	free(mat.csr_data);
}


